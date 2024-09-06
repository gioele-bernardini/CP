#include <iostream>

using namespace std;

class Vector2D {
private:
  double x, y;

public:
  // Constructor
  Vector2D(double _x = 0, double _y = 0) : x(_x), y(_y) {}

  // Overloading the + operator to add two vectors
  Vector2D operator+(const Vector2D& other) const {
    return Vector2D(x + other.x, y + other.y);
  }

  // Overloading the << operator for printing
  friend ostream& operator<<(ostream& os, const Vector2D& v) {
    os << "(" << v.x << ", " << v.y << ")";

    return os;
  }
};

int main() {
  Vector2D v1(2.5, 3.1);
  Vector2D v2(1.2, 2.8);

  // Adding two vectors using the overloaded + operator
  Vector2D v3 = v1 + v2;

  // Printing the result using the overloaded << operator
  cout << "Sum result: " << v3 << endl;

  return 0;
}

