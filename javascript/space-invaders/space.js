// Board
let tileSize = 32;
let rows = 16;
let columns = 16;

let board;
let boardWidth = tileSize * columns;
let boardHeight = tileSize * rows;
let context;

// Ship
let shipWidth = tileSize * 2;
let shipHeight = tileSize;
let shipX = tileSize * columns / 2 - tileSize;
let shipY = tileSize * rows - tileSize * 2;

let ship = {
  x : shipX,
  y : shipY,
  width : shipWidth,
  height : shipHeight,
}

let shipImg;
let shipVelocityX = tileSize;

window.onload = function() {
  board = document.getElementById('board');
  board.width = boardWidth;
  board.height = boardHeight;
  context = board.getContext('2d');

  // Draw initial ship
  // context.fillStyle = 'green';
  // context.fillRect(ship.x, ship.y, ship.width, ship.height);

  // Load images
  shipImg = new Image();
  shipImg.src = 'images/ship.png';
  shipImg.onload = function() {
    context.drawImage(shipImg, ship.x, ship.y, ship.width, ship.height);
  }

  requestAnimationFrame(update);
  document.addEventListener('keydown', moveShip);
}

function update() {
  requestAnimationFrame(update);

  context.clearRect(0, 0, board.width, board.height);

  // Ship
  context.drawImage(shipImg, ship.x, ship.y, ship.width, ship.height);
}

function moveShip(e) {
  if (e.code == 'KeyA' && ship.x - shipVelocityX >= 0) {
    ship.x -= shipVelocityX;
  }
  else if (e.code == 'KeyD' && ship.x + shipVelocityX + ship.width) {
    ship.x += shipVelocityX;
  }
}
