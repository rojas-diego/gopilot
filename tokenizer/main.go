// Here is a comment in first place
package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"io/ioutil"
	"log"
)

var (
	mode   *string
	config *string
)

func main() {
	mode = flag.String("mode", "", "mode to run in (either 'encode', 'decode' or 'train')")
	config = flag.String("config", "", "config file to use")
	flag.Parse()

	sampleArray := []string{"a", "b", "c"}
	fmt.Println(sampleArray)

	switch *mode {
	case "encode":
		encode()
	case "decode":
		decode()
	case "train":
		train()
	default:
		log.Fatal("Invalid mode, must be either 'encode', 'decode' or 'train'")
	}
}

// Config is the configuration for the tokenizer. Stored in a JSON file.
type Config struct {
	VocabSize     int      `json:"vocabSize"`
	SpecialTokens []string `json:"specialTokens"`
	Vocab         []string `json:"vocab"`
}

// Using the tokenizer config, encode the Go source code from the standard input
func encode() {
	// config := loadConfigOrPanic()
	log.Fatal("Not implemented")
}

// Using te tokenizer config, decode the raw tokenized data from the standard
// input
func decode() {
	// config := loadConfigOrPanic()
	log.Fatal("Not implemented")
}

// Train the tokenizer using the Go source code from the standard input
func train() {
	log.Println("Not implemented")
}

func loadConfigOrPanic() *Config {
	config, err := ioutil.ReadFile(*config)
	if err != nil {
		log.Fatal(err)
	}
	var c Config
	err = json.Unmarshal(config, &c)
	if err != nil {
		log.Fatal(err)
	}
	return &c
}
