#include <iostream>

// Base class 1
class Base1 {
public:
  void display() {
    std::cout << "Base1 display function." << std::endl;
  }
};

// Base class 2
class Base2 {
public:
  void display() {
    std::cout << "Base2 display function." << std::endl;
  }
};

// Derived class that inherits from Base1 and Base2
class Derived : public Base1, public Base2 {
public:
  // This function resolves ambiguity by explicitly calling Base1 or Base2 display
  void display() {
    Base1::display(); // Calls display from Base1
    Base2::display(); // Calls display from Base2
  }
};

int main() {
  Derived obj;
  obj.display();  // Calls the overridden display function in Derived class
  return 0;
}

