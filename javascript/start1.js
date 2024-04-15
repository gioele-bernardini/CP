console.log('Hello, world!');

// Is javascript an untyped language?
var myVar;  // Undefined (al momento!) (trivial)
console.log(typeof(myVar));

myVar = 'Pippo';
myVar = 5;
myVar = true;
myVar = [1, 2, 3];  // Composite/Non-Primitive type
myVar = {key1: 'value1'}; // Same here

myVar = null; // null type (trivial)
myVar = function(n) { return n + 1 };


///////////////////////////////////

var list = ['apple', 'pear', 'peach'];
list[0] // Accessing by id
list.indexOf('pear');
list.push('banana');
list.pop()
list.shift()  // Take the first element
list.length // NOT a function
list.slice(start, end);
list.join('separator'); // Same as python3


/////////////////////////////////////

for (let value of ['first', 'second']) {
  console.log(value);
}
[1, 2, 3].forEach( console.log )