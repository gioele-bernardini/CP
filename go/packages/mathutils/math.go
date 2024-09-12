package mathutils

// Exported: Can be used in other packages
func Add(a, b int) int {
	return a + b
}

// Not exported: Cannot be used in other packages
func subtract(a, b int) int {
	return a - b
}
