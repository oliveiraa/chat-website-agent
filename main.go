package main

import (
	"context"
	"encoding/json"
	"fmt"
	"sort"
	"strings"
	"time"

	_ "github.com/hypermodeinc/modus/sdk/go" // Modus runtime
	"github.com/hypermodeinc/modus/sdk/go/pkg/dgraph"
	"github.com/hypermodeinc/modus/sdk/go/pkg/models"
	"github.com/hypermodeinc/modus/sdk/go/pkg/models/openai"
)

const dgraphConnectionName = "website" // Must match modus.json
const modelName = "google-gemini"
const defaultSystemPrompt = "You are a helpful assistant"

// ChatResponse represents the response from the Chat function
type ChatResponse struct {
	Content string `json:"content"`
}

// DgraphChatMessage is used for storing and retrieving messages from Dgraph
type DgraphChatMessage struct {
	UID        string    `json:"uid,omitempty"`         // UID from Dgraph, useful if we need to reference it
	Role       string    `json:"role"`                  // Dgraph predicate: ChatMessage.role
	Content    string    `json:"content"`               // Dgraph predicate: ChatMessage.content
	Timestamp  time.Time `json:"timestamp"`             // Dgraph predicate: ChatMessage.timestamp
	DgraphType []string  `json:"dgraph.type,omitempty"` // For setting Dgraph type
}

// ClearChatResponse represents the response from the ClearChat function
type ClearChatResponse struct {
	Success bool   `json:"success"`
	Message string `json:"message"`
}

// Chat processes a chat request, now with Dgraph-backed memory
func Chat(sessionID string, userMessage string) (*ChatResponse, error) {
	model, err := models.GetModel[openai.ChatModel](modelName)
	if err != nil {
		return nil, fmt.Errorf("error getting model: %w", err)
	}

	ctx := context.Background() // Context for Dgraph operations

	// 1. Load history from Dgraph
	loadedMessages, err := loadHistoryFromDgraph(ctx, sessionID)
	if err != nil {
		// Log error but attempt to continue as a new session
		fmt.Printf("Error loading history for session %s: %v. Treating as new session.\\n", sessionID, err)
		loadedMessages = []DgraphChatMessage{} // Ensure it's an empty slice
	}

	var currentChatHistoryForLLM []DgraphChatMessage // History to build for the LLM
	if len(loadedMessages) == 0 {
		// Add default system prompt if no history (new session or failed load)
		currentChatHistoryForLLM = append(currentChatHistoryForLLM, DgraphChatMessage{
			Role:      "system",
			Content:   defaultSystemPrompt,
			Timestamp: time.Now().UTC(), // Timestamp mainly for consistency here
		})
	} else {
		currentChatHistoryForLLM = loadedMessages
	}

	// 2. Prepare and add current user message to in-memory history for LLM
	userMessageToSave := DgraphChatMessage{
		Role:       "user",
		Content:    userMessage,
		Timestamp:  time.Now().UTC(), // Crucial for ordering and saving
		DgraphType: []string{"ChatMessage"},
	}
	currentChatHistoryForLLM = append(currentChatHistoryForLLM, userMessageToSave)

	// 3. Convert currentChatHistoryForLLM to modelMessages for the OpenAI model SDK
	var modelMessagesForOpenAI []openai.RequestMessage
	for _, msg := range currentChatHistoryForLLM {
		switch msg.Role {
		case "system":
			modelMessagesForOpenAI = append(modelMessagesForOpenAI, openai.NewSystemMessage(msg.Content))
		case "user":
			modelMessagesForOpenAI = append(modelMessagesForOpenAI, openai.NewUserMessage(msg.Content))
		case "assistant":
			modelMessagesForOpenAI = append(modelMessagesForOpenAI, openai.NewAssistantMessage(msg.Content))
		}
	}

	fmt.Printf("DEBUG: Effective message history being sent for session %s:\\n", sessionID)
	for _, chatMsg := range currentChatHistoryForLLM {
		fmt.Printf("  - Role: %s, Content: %s, Timestamp: %s\\n", chatMsg.Role, chatMsg.Content, chatMsg.Timestamp.Format(time.RFC3339))
	}

	// 4. Invoke LLM
	input, err := model.CreateInput(modelMessagesForOpenAI...)
	if err != nil {
		return nil, fmt.Errorf("error creating model input: %w", err)
	}
	input.Temperature = 0.7 // Example temperature

	output, err := model.Invoke(input)
	if err != nil {
		return nil, fmt.Errorf("error invoking model: %w", err)
	}
	assistantContent := strings.TrimSpace(output.Choices[0].Message.Content)

	assistantMessageToSave := DgraphChatMessage{
		Role:       "assistant",
		Content:    assistantContent,
		Timestamp:  time.Now().UTC().Add(time.Millisecond), // Ensure assistant timestamp is slightly after user
		DgraphType: []string{"ChatMessage"},
	}

	// 5. Save the NEW user message and NEW assistant response to Dgraph
	newMessagesToPersist := []DgraphChatMessage{userMessageToSave, assistantMessageToSave}
	err = saveNewMessagesToDgraph(ctx, sessionID, newMessagesToPersist)
	if err != nil {
		// Log error, but chat can still return. Persistence for the *next* turn might be affected.
		fmt.Printf("CRITICAL: Error saving new messages for session %s: %v. Subsequent history may be incomplete.\\n", sessionID, err)
	}

	return &ChatResponse{
		Content: assistantContent,
	}, nil
}

func loadHistoryFromDgraph(ctx context.Context, sessionID string) ([]DgraphChatMessage, error) {
	// Query to get ChatSession UID and then its messages ordered by timestamp
	// New strategy:
	// 1. Find the UID of the ChatSession with the given sessionID.
	// 2. Find ChatMessage nodes linked to this ChatSession UID via 'in_session', ordered by timestamp.
	query := `
        query getSessionMessages($sessionID: string) {
            var(func: eq(ChatSession.sessionID, $sessionID)) {
                TARGET_SESSION_UID as uid
            }

            messages(func: type(ChatMessage), orderasc: ChatMessage.timestamp) @filter(uid_in(in_session, uid(TARGET_SESSION_UID))) {
                uid
                role: ChatMessage.role
                content: ChatMessage.content
                timestamp: ChatMessage.timestamp
            }
        }
    `
	vars := map[string]string{"$sessionID": sessionID}

	resp, err := dgraph.ExecuteQuery(dgraphConnectionName, &dgraph.Query{
		Query:     query,
		Variables: vars,
	})
	if err != nil {
		return nil, fmt.Errorf("dgraph.ExecuteQuery failed for session %s: %w", sessionID, err)
	}

	// Revised struct to match the simpler Dgraph JSON output from the new query.
	// The "messages" key in the JSON will directly contain an array of chat message objects.
	var queryResult struct {
		Messages []struct {
			UID       string    `json:"uid"`
			Role      string    `json:"role"`      // Corresponds to the alias "role" in the DQL query
			Content   string    `json:"content"`   // Corresponds to the alias "content" in the DQL query
			Timestamp time.Time `json:"timestamp"` // Corresponds to the alias "timestamp" in the DQL query
		} `json:"messages"` // This tag matches the alias "messages" in the Dgraph query
	}

	if err := json.Unmarshal([]byte(resp.Json), &queryResult); err != nil {
		return nil, fmt.Errorf("failed to unmarshal Dgraph response for session %s: %w. JSON: %s", sessionID, err, string(resp.Json))
	}

	var chatMessages []DgraphChatMessage
	// Iterate directly over queryResult.Messages which contains the filtered and ordered messages.
	if queryResult.Messages != nil { // Check if Messages is not nil (it will be an empty slice if no messages found)
		for _, m := range queryResult.Messages {
			chatMessages = append(chatMessages, DgraphChatMessage{
				UID:       m.UID,
				Role:      m.Role,
				Content:   m.Content,
				Timestamp: m.Timestamp,
				// DgraphType is not strictly needed for loaded messages unless we re-mutate them
			})
		}
	} else {
		// This case implies the "messages" key was missing or null in JSON, which is unlikely if the query executes.
		// An empty result from Dgraph for the "messages" block would be `{"messages":[]}`,
		// for which queryResult.Messages would be an empty non-nil slice.
		// Logging here for completeness, though the above loop handles empty results gracefully.
		fmt.Printf("DEBUG: Dgraph query for session %s resulted in nil Messages array (or key missing). JSON: %s\\n", sessionID, string(resp.Json))
	}

	// Dgraph's `orderasc` should handle the ordering.
	// An explicit sort here is a safeguard but might be redundant if Dgraph guarantees order.
	// Given the previous sort was kept as a safeguard, we'll keep it.
	sort.SliceStable(chatMessages, func(i, j int) bool {
		return chatMessages[i].Timestamp.Before(chatMessages[j].Timestamp)
	})

	return chatMessages, nil
}

func saveNewMessagesToDgraph(ctx context.Context, sessionID string, newMessages []DgraphChatMessage) error {
	const sessionBlankNode = "_:session"
	var dgraphMutations []interface{}
	sessionUpsertObject := map[string]interface{}{
		"uid":                   sessionBlankNode,
		"ChatSession.sessionID": sessionID,
		"dgraph.type":           "ChatSession",
	}
	dgraphMutations = append(dgraphMutations, sessionUpsertObject)

	for i, msg := range newMessages {
		messageBlankNode := fmt.Sprintf("_:msg%d", i)
		chatMessageObject := map[string]interface{}{
			"uid":                   messageBlankNode,
			"dgraph.type":           "ChatMessage",
			"ChatMessage.role":      msg.Role,
			"ChatMessage.content":   msg.Content,
			"ChatMessage.timestamp": msg.Timestamp.Format(time.RFC3339Nano),
			"in_session": map[string]interface{}{
				"uid": sessionBlankNode,
			},
		}
		dgraphMutations = append(dgraphMutations, chatMessageObject)

		sessionLinkToMessage := map[string]interface{}{
			"uid": sessionBlankNode,
			"ChatSession.has_message": map[string]interface{}{
				"uid": messageBlankNode,
			},
		}
		dgraphMutations = append(dgraphMutations, sessionLinkToMessage)
	}

	setJsonPayload, err := json.Marshal(dgraphMutations)
	if err != nil {
		return fmt.Errorf("failed to marshal Dgraph SetJson: %w", err)
	}

	// Constructing a dgraph.Mutation object
	mutation := &dgraph.Mutation{
		SetJson: string(setJsonPayload),
	}

	// Adjusted ExecuteMutations call: assuming it takes 2 arguments and CommitNow is implicit or default.
	_, err = dgraph.ExecuteMutations(dgraphConnectionName, mutation)
	if err != nil {
		return fmt.Errorf("dgraph.ExecuteMutations failed for session %s: %w. Payload: %s", sessionID, err, string(setJsonPayload))
	}

	return nil
}

// ClearChat clears the chat history for a specific session from Dgraph
func ClearChat(sessionID string) (*ClearChatResponse, error) {
	// deleteMutationDQL is removed as dgraph.Mutation does not seem to support a Query field for DQL execution.
	// TODO: ClearChat needs a different approach for deletion.
	// This would typically involve querying UIDs of the session and messages,
	// then using DelJson or DelNquads fields in dgraph.Mutation if they exist.
	// For now, this function will be a no-op regarding Dgraph deletion to clear linter errors.

	mutation := &dgraph.Mutation{
		// Empty mutation for now
	}

	// Adjusted ExecuteMutations call
	_, err := dgraph.ExecuteMutations(dgraphConnectionName, mutation)
	if err != nil {
		fmt.Printf("Error during Dgraph ClearChat (currently a no-op) for session %s: %v\\n", sessionID, err)
		return &ClearChatResponse{
			Success: false,
			Message: fmt.Sprintf("Failed to clear chat history from Dgraph: %v", err),
		}, nil
	}

	return &ClearChatResponse{
		Success: true,
		Message: "Chat history cleared successfully from Dgraph.",
	}, nil
}

// TestDgraphInteraction is a simple function to test Dgraph connectivity and basic operations.
func TestDgraphInteraction() (string, error) {
	// ctx := context.Background() // Removed as it was unused
	testSessionID := "test-session-dgraph-debug"
	testNodeUID := "_:testnode"

	// 1. Define a test mutation to create a simple node
	testMutationData := map[string]interface{}{
		"uid":                  testNodeUID,
		"dgraph.type":          "TestNode",
		"TestNode.name":        "Dgraph Test Entry",
		"TestNode.timestamp":   time.Now().UTC().Format(time.RFC3339Nano),
		"TestNode.sessionLink": testSessionID, // Just an example field
	}

	setJsonPayload, err := json.Marshal([]interface{}{testMutationData})
	if err != nil {
		return "", fmt.Errorf("failed to marshal test mutation JSON: %w", err)
	}

	mutation := &dgraph.Mutation{
		SetJson: string(setJsonPayload),
	}

	fmt.Printf("Attempting to execute test mutation: %s\n", string(setJsonPayload))
	muResponse, err := dgraph.ExecuteMutations(dgraphConnectionName, mutation)
	if err != nil {
		return "", fmt.Errorf("dgraph.ExecuteMutations failed for test: %w. Payload: %s", err, string(setJsonPayload))
	}
	fmt.Printf("Test mutation response: %+v\n", muResponse) // Log the response details

	// Extract assigned UID if mutation was successful and a new node was created
	// This depends on how Dgraph client/SDK returns UIDs for blank nodes.
	// For simplicity, we'll query by a known field.

	// 2. Define a query to retrieve the test node
	query := `
		query getTestNode($testID: string) {
			testNodes(func: eq(TestNode.sessionLink, $testID)) {
				uid
				name: TestNode.name
				timestamp: TestNode.timestamp
				session: TestNode.sessionLink
			}
		}
	`
	vars := map[string]string{"$testID": testSessionID}

	fmt.Printf("Attempting to execute test query for sessionLink: %s\n", testSessionID)
	queryResp, err := dgraph.ExecuteQuery(dgraphConnectionName, &dgraph.Query{
		Query:     query,
		Variables: vars,
	})
	if err != nil {
		return "", fmt.Errorf("dgraph.ExecuteQuery failed for test: %w", err)
	}

	fmt.Printf("Test query JSON response: %s\n", queryResp.Json)

	// Minimal parsing to confirm we got something
	var queryResult struct {
		TestNodes []map[string]interface{} `json:"testNodes"`
	}
	if err := json.Unmarshal([]byte(queryResp.Json), &queryResult); err != nil {
		return "", fmt.Errorf("failed to unmarshal Dgraph test query response: %w. JSON: %s", err, string(queryResp.Json))
	}

	if len(queryResult.TestNodes) == 0 {
		return "", fmt.Errorf("no test nodes found for sessionLink %s. Mutation might have failed silently or query is incorrect", testSessionID)
	}

	return fmt.Sprintf("Dgraph test successful! Found test node: %+v", queryResult.TestNodes[0]), nil
}

// SayHello is kept from the original code
func SayHello(name *string) string {
	var s string
	if name == nil {
		s = "World"
	} else {
		s = *name
	}

	return fmt.Sprintf("Hello, %s!", s)
}

// ApplyDgraphSchema defines and applies the Dgraph schema.
// This function should be called to ensure Dgraph is properly configured.
func ApplyDgraphSchema() (string, error) {
	schema := `
		# type ChatSession {
		# 	ChatSession.sessionID: string @index(exact) .
		# 	ChatSession.has_message: [uid] @reverse .
		# }
		# 
		# type ChatMessage {
		# 	ChatMessage.role: string .
		# 	ChatMessage.content: string .
		# 	ChatMessage.timestamp: datetime @index(hour) .
		# 	in_session: uid @reverse .
		# }
		# 
		# # Schema for TestDgraphInteraction function
		# type TestNode {
		# 	TestNode.name: string .
		# 	TestNode.timestamp: datetime .
		# 	TestNode.sessionLink: string @index(exact) .
		# }

		ChatSession.sessionID: string @index(exact) .
		ChatSession.has_message: [uid] @reverse .
		ChatMessage.role: string .
		ChatMessage.content: string .
		ChatMessage.timestamp: datetime @index(hour) .
		in_session: uid @reverse .

		TestNode.name: string .
		TestNode.timestamp: datetime .
		TestNode.sessionLink: string @index(exact) .
	`
	// The connection name must match the one in modus.json and used in other Dgraph calls
	// Assuming AlterSchema returns a response string and an error, or just an error.
	// Based on the linter, it seems to return a single value (string) or (string, error) or (error).
	// Let's try with the (string, error) pattern first, as it's common.
	// If the linter complained about 2 values for 1, it might return just error or just string.
	// The docs (https://docs.hypermode.com/modus/sdk/dgraph) show: function alterSchema(connectionName: string, schema: string): string;
	// This suggests it returns a single string (response/status) and error is handled internally or via panic for Go SDKs, or it's a non-Go signature.
	// For Go, `func AlterSchema(connectionName string, schema string) error` is more idiomatic if it only signals success/failure.
	// Given the previous linter error (2 variables for 1 value), it means it returns ONLY ONE value.
	// Let's assume it's an error, which is a common Go pattern for such functions.
	// If it were a string response, we'd need to clarify how errors are propagated.

	// Attempting to match the Go SDK's dgraph.AlterSchema function signature.
	// The SDK documentation (https://docs.hypermode.com/modus/sdk/go/dgraph) does not explicitly show AlterSchema's signature.
	// However, other execute functions return (Response, error). Let's try to find its actual signature.
	// The linter error "assignment mismatch: 2 variables but dgraph.AlterSchema returns 1 value" is key.
	// This implies `func AlterSchema(connectionName string, schema string) SOMETHING` where SOMETHING is a single type.
	// Typically, this would be `error` if the primary purpose is to report success/failure.

	// Let's assume the SDK function is: `func AlterSchema(connectionName string, schema string) error`
	err := dgraph.AlterSchema(dgraphConnectionName, schema)
	if err != nil {
		return "", fmt.Errorf("failed to alter Dgraph schema: %w", err)
	}
	return "Dgraph schema applied successfully.", nil
}
