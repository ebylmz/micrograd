from micrograd.engine import Value

# Define scalar inputs
a = Value(-4.0)
b = Value(2.0)

# Computational graph forward pass
c = a + b
d = a * b + b**3
c += c + 1
c += 1 + c + (-a)
d += d * 2 + (b + a).relu()
d += 3 * d + (b - a).relu()
e = c - d
f = e**2
g = f / 2.0
g += 10.0 / f

print(f"Forward outcome (g.data): {g.data:.4f}")  # Outputs 24.7041

# Backpropagate through the graph
g.backward()

print(f"Gradient dg/da: {a.grad:.4f}")  # Outputs 138.8338
print(f"Gradient dg/db: {b.grad:.4f}")  # Outputs 645.5773