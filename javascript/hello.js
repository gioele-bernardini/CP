// This file covers the basics of the first lecture

console.log('Hello, world!');


var myVar;
console.log(typeof (myVar));

myVar = 'Pippo';
myVar = 5;
myVar = true;
myVar = [1, 2, 3];
myVar = {key1: 'value1'};
myVar = null;

myVar = function(n) { return n + 1 };


for (let value of ['first', 'second']) {
  console.log(value)
}
[1, 2, 3].forEach( console.log )