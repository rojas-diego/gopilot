package main

import (
	"fmt"
	"go/ast"
	"go/parser"
	"go/token"
	"io/ioutil"
	"log"
	"os"
	"reflect"
	"strings"
)

type Node struct {
	Start int
	End   int
	Node  ast.Node
	Name  string
}

func main() {
	file, err := ioutil.ReadAll(os.Stdin)
	if err != nil {
		log.Fatal(err)
	}

	fset := token.NewFileSet()
	parsedFile, err := parser.ParseFile(fset, "", file, parser.ParseComments)
	if err != nil {
		log.Println(err)
	}

	nodes := make([]Node, 0)

	ast.Inspect(parsedFile, func(n ast.Node) bool {
		if n == nil {
			return false
		}

		nodes = append(nodes, Node{
			Start: fset.Position(n.Pos()).Offset,
			End:   fset.Position(n.End()).Offset,
			Node:  n,
			Name:  reflect.TypeOf(n).Elem().Name(),
		})

		return true
	})

	// Sort the nodes so that start pos is ascending
	for i := 0; i < len(nodes); i++ {
		for j := i + 1; j < len(nodes); j++ {
			if nodes[i].Start > nodes[j].Start {
				tmp := nodes[i]
				nodes[i] = nodes[j]
				nodes[j] = tmp
			}
		}
	}

	println("--- Nodes ---")
	printNodes(nodes)

	println("--- Tree ---")
	printTree(nodes, 0, 0)
}

func printNodes(nodes []Node) {
	for _, node := range nodes {
		fmt.Println(node.Name + " (" + fmt.Sprint(node.Start) + ", " + fmt.Sprint(node.End) + ")")
	}
}

func printTree(nodes []Node, index int, depth int) int {
	if index >= len(nodes) {
		return index
	}

	node := nodes[index]
	numChildren := 0
	for tmpIndex := index + 1; tmpIndex < len(nodes); tmpIndex++ {
		child := nodes[tmpIndex]
		if child.Start >= node.Start && child.End <= node.End {
			numChildren++
		} else {
			break
		}
	}

	if numChildren == 0 {
		fmt.Println(strings.Repeat("  ", depth) + node.Name + " (" + fmt.Sprint(node.Start) + ", " + fmt.Sprint(node.End) + ") value=" + fmt.Sprint(node.Node))
	} else {
		fmt.Println(strings.Repeat("  ", depth) + node.Name + " (" + fmt.Sprint(node.Start) + ", " + fmt.Sprint(node.End) + ")")
	}

	// If the node has children, print them
	index = index + 1
	for index < len(nodes) {
		child := nodes[index]
		if child.Start >= node.Start && child.End <= node.End {
			index = printTree(nodes, index, depth+1)
			numChildren++
		} else {
			break
		}
	}

	// fmt.Println("index:", index, ", depth: ", depth)

	if depth == 0 {
		// If there are other nodes on the same level, print them
		return printTree(nodes, index, depth)
	}

	return index
}
