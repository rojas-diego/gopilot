package main

import "C"
import (
	"encoding/json"
	"log"
)

type Config struct {
	VocabSize int `json:"vocab_size"`
}

type Tokenizer struct {
	Config *Config
}

type EncodeArgs struct {
	Sequence string `json:"sequence"`
}

type EncodeResult struct {
	IDs               []int   `json:"ids"`
	Offsets           [][]int `json:"offsets"`
	AttentionMask     []int   `json:"attention_mask"`
	SpecialTokensMask []int   `json:"special_tokens_mask"`
}

type DecodeArgs struct {
	IDs []int `json:"ids"`
}

type TrainArgs struct {
	Files []string `json:"files"`
}

func NewTokenizer() *Tokenizer {
	return &Tokenizer{}
}

func NewTokenizerFromConfig(config *Config) *Tokenizer {
	return &Tokenizer{
		Config: config,
	}
}

func (t *Tokenizer) Encode(args *EncodeArgs) *EncodeResult {
	return &EncodeResult{
		IDs: []int{1, 2, 3},
		Offsets: [][]int{
			{1, 2},
			{2, 3},
			{3, 4},
		},
		AttentionMask:     []int{1, 1, 1},
		SpecialTokensMask: []int{0, 0, 0},
	}
}

func (t *Tokenizer) Decode(args *DecodeArgs) string {
	return "Hello World!"
}

func (t *Tokenizer) ToJSON() string {
	configJsonBytes, err := json.Marshal(t.Config)
	if err != nil {
		return ""
	}

	return string(configJsonBytes)
}

func (t *Tokenizer) Train(args *TrainArgs) error {
	return nil
}

//export Encode
func Encode(configJson *C.char, encodeArgsJson *C.char) *C.char {
	config := &Config{}
	err := json.Unmarshal([]byte(C.GoString(configJson)), config)
	if err != nil {
		return nil
	}

	tokenizer := NewTokenizerFromConfig(config)

	encodeArgs := &EncodeArgs{}
	err = json.Unmarshal([]byte(C.GoString(encodeArgsJson)), encodeArgs)
	if err != nil {
		log.Println(err)
		return nil
	}

	encodeResult := tokenizer.Encode(encodeArgs)
	encodeResultJsonBytes, err := json.Marshal(encodeResult)
	if err != nil {
		log.Println(err)
		return nil
	}

	return C.CString(string(encodeResultJsonBytes))
}

//export Decode
func Decode(configJson *C.char, decodeArgsJson *C.char) *C.char {
	config := &Config{}
	err := json.Unmarshal([]byte(C.GoString(configJson)), config)
	if err != nil {
		return nil
	}

	tokenizer := NewTokenizerFromConfig(config)

	decodeArgs := &DecodeArgs{}
	err = json.Unmarshal([]byte(C.GoString(decodeArgsJson)), decodeArgs)
	if err != nil {
		return nil
	}

	sequence := tokenizer.Decode(decodeArgs)

	return C.CString(sequence)
}

//export Train
func Train(trainArgsJson *C.char) *C.char {
	trainArgs := &TrainArgs{}
	err := json.Unmarshal([]byte(C.GoString(trainArgsJson)), trainArgs)
	if err != nil {
		return nil
	}

	tokenizer := NewTokenizer()

	err = tokenizer.Train(trainArgs)
	if err != nil {
		return nil
	}

	return C.CString(tokenizer.ToJSON())
}

func main() {}
