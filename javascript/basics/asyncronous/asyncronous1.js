// Sincrono: prima legge e poi termina (sequenziale!)
var fs = require('fs');
var data = fs.readFileSync('file.txt', 'utf-8');

console.log(data);
console.log('File read :)');


// Asincrono: (termina prima, ma poi riceve il callback!)
fs.readFile('file.txt', 'utf-8', function(error, data) {
  console.log(data);
});

console.log('File read once more :)');