package main

func main() {
	if 1+3 == 4 {
		for i := 0; i < 10; i++ {
			if i%2 == 0 {
				return
			}
		}
	}
}
