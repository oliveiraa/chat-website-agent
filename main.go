package main

import (
	"fmt"
	"strings"

	_ "github.com/hypermodeinc/modus/sdk/go"
	"github.com/hypermodeinc/modus/sdk/go/pkg/models"
	"github.com/hypermodeinc/modus/sdk/go/pkg/models/openai"
)

// The model name defined in modus.json
const modelName = "google-gemini"

// ChatResponse represents the response from the Chat function
type ChatResponse struct {
	Content string `json:"content"`
}

// Chat processes a chat request with Gemini 2.5 Flash
// This function will be automatically exposed as an endpoint by Modus
func Chat(systemPrompt string, userMessage string) (*ChatResponse, error) {
	model, err := models.GetModel[openai.ChatModel](modelName)
	if err != nil {
		return nil, fmt.Errorf("error getting model: %w", err)
	}

	input, err := model.CreateInput(
		openai.NewSystemMessage(systemPrompt),
		openai.NewUserMessage(userMessage),
	)
	if err != nil {
		return nil, fmt.Errorf("error creating model input: %w", err)
	}

	// Set temperature for some controlled randomness
	input.Temperature = 0.7

	output, err := model.Invoke(input)
	if err != nil {
		return nil, fmt.Errorf("error invoking model: %w", err)
	}

	return &ChatResponse{
		Content: strings.TrimSpace(output.Choices[0].Message.Content),
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