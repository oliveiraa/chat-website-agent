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

	turnTimestamp := time.Now().UTC() // Capture timestamp for the current turn

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
		Timestamp:  turnTimestamp, // Use captured turn timestamp
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
		Timestamp:  turnTimestamp, // Use captured turn timestamp
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
	// 1. Find the UID of the ChatSession with the given sessionID.
	// 2. Find ChatMessage nodes linked to this ChatSession via the new ChatMessage.sessionIDRef predicate, ordered by timestamp.
	query := `
        query getSessionMessages($sessionID: string) {
            messages(func: eq(ChatMessage.sessionIDRef, $sessionID), orderasc: ChatMessage.timestamp) @filter(type(ChatMessage)) {
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
			"uid":                      messageBlankNode,
			"dgraph.type":              "ChatMessage",
			"ChatMessage.role":         msg.Role,
			"ChatMessage.content":      msg.Content,
			"ChatMessage.timestamp":    msg.Timestamp.Format(time.RFC3339Nano),
			"ChatMessage.sessionIDRef": sessionID, // Link message to session by sessionID
		}
		dgraphMutations = append(dgraphMutations, chatMessageObject)
		// The explicit sessionLinkToMessage mutation is no longer needed
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
	// 1. Query for UIDs of the session and its messages
	query := `
        query getUidsForDeletion($sessionID: string) {
            session(func: eq(ChatSession.sessionID, $sessionID)) @filter(type(ChatSession)) {
                uid
            }
            messages(func: eq(ChatMessage.sessionIDRef, $sessionID)) @filter(type(ChatMessage)) {
                uid
            }
        }
    `
	vars := map[string]string{"$sessionID": sessionID}

	queryResponse, err := dgraph.ExecuteQuery(dgraphConnectionName, &dgraph.Query{
		Query:     query,
		Variables: vars,
	})
	if err != nil {
		return &ClearChatResponse{
			Success: false,
			Message: fmt.Sprintf("Failed to query Dgraph for UIDs to delete session %s: %v", sessionID, err),
		}, nil
	}

	var queryResult struct {
		Session []struct {
			UID string `json:"uid"`
		} `json:"session"`
		Messages []struct {
			UID string `json:"uid"`
		} `json:"messages"`
	}
	if err := json.Unmarshal([]byte(queryResponse.Json), &queryResult); err != nil {
		return &ClearChatResponse{
			Success: false,
			Message: fmt.Sprintf("Failed to unmarshal Dgraph response for session %s UIDs: %v. JSON: %s", sessionID, err, string(queryResponse.Json)),
		}, nil
	}

	// 2. Collect UIDs for deletion
	var uidsToDelete []string // Store UIDs directly as strings
	if len(queryResult.Session) > 0 && queryResult.Session[0].UID != "" {
		uidsToDelete = append(uidsToDelete, queryResult.Session[0].UID)
	}
	for _, msg := range queryResult.Messages {
		if msg.UID != "" {
			uidsToDelete = append(uidsToDelete, msg.UID)
		}
	}

	if len(uidsToDelete) == 0 {
		return &ClearChatResponse{
			Success: true,
			Message: fmt.Sprintf("No chat history found for session %s, or it was already clear.", sessionID),
		}, nil
	}

	// 3. Format UIDs into N-Quad delete statements
	var nquadsBuilder strings.Builder
	for _, uid := range uidsToDelete {
		nquadsBuilder.WriteString(fmt.Sprintf("<%s> * * .\n", uid))
	}
	deleteNquadsPayload := nquadsBuilder.String()

	// 4. Create and execute mutation for deletion using DelNquads
	// This assumes the dgraph.Mutation struct supports a `DelNquads` field.
	mutation := &dgraph.Mutation{
		DelNquads: deleteNquadsPayload, // Using DelNquads
	}

	_, err = dgraph.ExecuteMutations(dgraphConnectionName, mutation)
	if err != nil {
		return &ClearChatResponse{
			Success: false,
			Message: fmt.Sprintf("Dgraph ExecuteMutations failed to delete data for session %s using N-Quads: %v. Payload:\n%s", sessionID, err, deleteNquadsPayload),
		}, nil
	}

	return &ClearChatResponse{
		Success: true,
		Message: fmt.Sprintf("Chat history for session %s (including %d messages and the session node) cleared successfully from Dgraph.", sessionID, len(queryResult.Messages)),
	}, nil
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
		ChatSession.sessionID: string @index(exact) .
		ChatMessage.role: string .
		ChatMessage.content: string .
		ChatMessage.timestamp: datetime @index(day) @index(hour) .
		ChatMessage.sessionIDRef: string @index(exact) .
	`
	// The connection name must match the one in modus.json and used in other Dgraph calls
	err := dgraph.AlterSchema(dgraphConnectionName, schema)
	if err != nil {
		return "", fmt.Errorf("failed to alter Dgraph schema: %w", err)
	}
	return "Dgraph schema applied successfully.", nil
}
