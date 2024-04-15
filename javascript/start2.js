// How to define a function?

function add(a, b) {
  return a + b;
};

var mult = function(a, b) {
  return a * b;
}

var mod = (a, b) => {
  let r = a % b;
  return r;
}

var pow2 = a => a * a

/*
add (1, 2)
mult(1, 2)
arrowFunction(4, 2)
*/

// Cosa e' il call-back?
[1, 2, 3].forEach(element => {
  console.log(element);
});


////////////////////////////////

Scope of variables (& constants): global, block, function
const global_const = 'global_const'; // globally-scoped constant
var global_var = 'global_var'; // globally-scoped variable
function myFn () {
console.log(global_const) // 'global_const'
console.log(global_var) // 'global_var'
if ( true ) {
const constant = 'constant'; // block-scoped constant
let local = 'local'; // block-scoped (local) variable
var variable = 'variable'; // function-scoped variable
}
console.log(constant) // ReferenceError
console.log(local) // ReferenceError
console.log(variable) // 'variable'
}
console.log(variable) // ReferenceError