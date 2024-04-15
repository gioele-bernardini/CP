var car = {
  type : 'Fiat',
  model : '500',
  
  description : function() {
    return this.type + ', ' + this.model;
  }
}

console.log(car); // Stampa l'intera struttura!
console.log(car.description()); // Usa .description()

// Pattern per simulare classe da funzione
function Car(type, model, color) {
  this.type = type;
  this.model = model;
  this.color = color;

  this.description = function() {
    return this.color + ' ' + this.model + ', ' + this.type;
  }
}

var fiat500rossa = new Car('Fiat', '500', 'rossa');
console.log(fiat500rossa);
console.log(fiat500rossa.description());


// Pattern per simulare ereditarieta'
let animal = {
  eats : true,
  walk() { console.log('Animal walk'); }
};
let rabbit = {
  jumps : true,
  __proto__ : animal,
};

console.log( rabbit.eats );
console.log( rabbit.jumps );
rabbit.walk();


// Using 'class' construct from ES6
class Car3 {
constructor(type, model, color) {
this.type = type;
this.model = model;
this.color = color;
}
description() {
return this.color + ", " + this.model + ", " + this.type;
};
}
var fiatPuntobianca = new Car3('Fiat', 'Punto', 'white');
console.log(fiatPuntobianca);
console.log(fiatPuntobianca.description());

class Suv extends Car3 {
description() {
return this.color + ", " + this.model + ", " + this.type + ", SUV";
};
}
var NissanQuashqai = new Suv('Nissan', 'Quashqai', 'black');
console.log(NissanQuashqai);
console.log(NissanQuashqai.description());