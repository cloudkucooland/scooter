package rknn

import (
	"bufio"
	"encoding/json"
	"fmt"
	"math"
	"os"
	"regexp"
	"strings"
	"sync"
)

const (
	clipMaxLength       = 77
	clipBOSID     int32 = 49406 // <|startoftext|>
	clipEOSID     int32 = 49407 // <|endoftext|>
)

// CLIP's pre-tokenization regex (equivalent to OpenAI's pat without negative lookahead)
var clipTokenPat = regexp.MustCompile(`(?i)<\|startoftext\|>|<\|endoftext\|>|'s|'t|'re|'ve|'m|'ll|'d|[\p{L}]+|[\p{N}]|[^\s\p{L}\p{N}]+`)

type Tokenizer struct {
	vocab       map[string]int32
	revVocab    map[int32]string
	merges      map[string]int32
	byteEncoder map[byte]rune
	byteDecoder map[rune]byte
	cache       sync.Map // word string -> []string BPE tokens
	unkID       int32
}

// NewTokenizer loads CLIP vocab.json and merges.txt.
func NewTokenizer(vocabPath, mergesPath string) (*Tokenizer, error) {
	vocabBytes, err := os.ReadFile(vocabPath)
	if err != nil {
		return nil, fmt.Errorf("read vocab: %w", err)
	}

	vocab := make(map[string]int32)
	if err := json.Unmarshal(vocabBytes, &vocab); err != nil {
		return nil, fmt.Errorf("parse vocab json: %w", err)
	}

	revVocab := make(map[int32]string, len(vocab))
	for k, v := range vocab {
		revVocab[v] = k
	}

	mergesFile, err := os.Open(mergesPath)
	if err != nil {
		return nil, fmt.Errorf("open merges: %w", err)
	}
	defer mergesFile.Close()

	merges := make(map[string]int32)
	scanner := bufio.NewScanner(mergesFile)
	rank := int32(0)
	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if line == "" || strings.HasPrefix(line, "#version") {
			continue
		}
		merges[line] = rank
		rank++
	}
	if err := scanner.Err(); err != nil {
		return nil, fmt.Errorf("scan merges: %w", err)
	}

	unk := clipEOSID
	if id, ok := vocab["<unk>"]; ok {
		unk = id
	}

	b2u, u2b := buildBytesToUnicode()

	return &Tokenizer{
		vocab:       vocab,
		revVocab:    revVocab,
		merges:      merges,
		byteEncoder: b2u,
		byteDecoder: u2b,
		unkID:       unk,
	}, nil
}

// Encode produces a 77-token slice padded with EOS (49407) for text_encoder 1.
func (t *Tokenizer) Encode(text string) ([]int32, error) {
	return t.encodeWithPad(text, clipEOSID), nil
}

// EncodeDual produces both 77-token inputs required by SDXL / LCM-SSD-1B:
// ids1 pads with 49407 (CLIP-ViT-L), ids2 pads with 0 (OpenCLIP-ViT-bigG).
func (t *Tokenizer) EncodeDual(text string) (ids1 []int32, ids2 []int32, err error) {
	return t.encodeWithPad(text, clipEOSID), t.encodeWithPad(text, 0), nil
}

func (t *Tokenizer) encodeWithPad(text string, padID int32) []int32 {
	// Normalize whitespace and lowercase per CLIPTokenizer
	cleaned := strings.ToLower(strings.Join(strings.Fields(text), " "))

	tokens := make([]int32, clipMaxLength)
	for i := range tokens {
		tokens[i] = padID
	}
	tokens[0] = clipBOSID

	pos := 1
	matches := clipTokenPat.FindAllString(cleaned, -1)
	for _, m := range matches {
		if pos >= clipMaxLength-1 {
			break
		}
		// Map UTF-8 bytes through OpenAI byte-level unicode table
		var sb strings.Builder
		for i := 0; i < len(m); i++ {
			sb.WriteRune(t.byteEncoder[m[i]])
		}
		bpeTokens := t.bpe(sb.String())
		for _, tok := range bpeTokens {
			if pos >= clipMaxLength-1 {
				break
			}
			if id, ok := t.vocab[tok]; ok {
				tokens[pos] = id
			} else {
				tokens[pos] = t.unkID
			}
			pos++
		}
	}

	// Place EOS token immediately after the last prompt token
	tokens[pos] = clipEOSID
	return tokens
}

func (t *Tokenizer) bpe(token string) []string {
	if cached, ok := t.cache.Load(token); ok {
		return cached.([]string)
	}

	runes := []rune(token)
	if len(runes) == 0 {
		return nil
	}

	// CLIP appends </w> to the final character of each word
	word := make([]string, len(runes))
	for i := 0; i < len(runes)-1; i++ {
		word[i] = string(runes[i])
	}
	word[len(runes)-1] = string(runes[len(runes)-1]) + "</w>"

	if len(word) == 1 {
		t.cache.Store(token, word)
		return word
	}

	for {
		bestRank := int32(math.MaxInt32)
		bestFirst, bestSecond := "", ""
		found := false

		for i := 0; i < len(word)-1; i++ {
			pair := word[i] + " " + word[i+1]
			if rank, ok := t.merges[pair]; ok && rank < bestRank {
				bestRank = rank
				bestFirst = word[i]
				bestSecond = word[i+1]
				found = true
			}
		}

		if !found {
			break
		}

		merged := make([]string, 0, len(word))
		for i := 0; i < len(word); {
			if i < len(word)-1 && word[i] == bestFirst && word[i+1] == bestSecond {
				merged = append(merged, bestFirst+bestSecond)
				i += 2
			} else {
				merged = append(merged, word[i])
				i++
			}
		}
		word = merged
		if len(word) == 1 {
			break
		}
	}

	t.cache.Store(token, word)
	return word
}

func (t *Tokenizer) Decode(ids []int32) string {
	var sb strings.Builder
	for _, id := range ids {
		if id == clipBOSID || id == clipEOSID || id == 0 {
			continue
		}
		if tok, ok := t.revVocab[id]; ok {
			sb.WriteString(tok)
		}
	}
	// Convert byte-level unicode runes back to UTF-8 bytes and replace </w> with spaces
	raw := sb.String()
	buf := make([]byte, 0, len(raw))
	for _, r := range raw {
		if b, ok := t.byteDecoder[r]; ok {
			buf = append(buf, b)
		}
	}
	return strings.TrimSpace(strings.ReplaceAll(string(buf), "</w>", " "))
}

// buildBytesToUnicode replicates OpenAI CLIP's bytes_to_unicode() mapping table.
func buildBytesToUnicode() (map[byte]rune, map[rune]byte) {
	bs := make([]int, 0, 256)
	for b := int('!'); b <= int('~'); b++ {
		bs = append(bs, b)
	}
	for b := 0xA1; b <= 0xAC; b++ {
		bs = append(bs, b)
	}
	for b := 0xAE; b <= 0xFF; b++ {
		bs = append(bs, b)
	}

	inBS := make(map[int]bool, len(bs))
	for _, b := range bs {
		inBS[b] = true
	}

	cs := make([]int, len(bs))
	copy(cs, bs)

	n := 0
	for b := 0; b < 256; b++ {
		if !inBS[b] {
			bs = append(bs, b)
			cs = append(cs, 256+n)
			n++
		}
	}

	b2u := make(map[byte]rune, 256)
	u2b := make(map[rune]byte, 256)
	for i := range bs {
		b2u[byte(bs[i])] = rune(cs[i])
		u2b[rune(cs[i])] = byte(bs[i])
	}
	return b2u, u2b
}
