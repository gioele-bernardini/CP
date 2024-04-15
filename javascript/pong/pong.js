// Board properties
let board;
let boardWidth = 900;
let boardHeight = 900;
let context;

// Players
let playerWidth = 20;
let playerHeight = 100;
let playerVelocityY = 0;

let player1 = {
  x  : 10,
  y : boardHeight / 2,
  width : playerWidth,
  height : playerHeight,
  velocityY : playerVelocityY,
}

let player2 = {
  x : boardWidth - playerWidth - 10,
  y : boardHeight / 2,
  width : playerWidth,
  height : playerHeight,
  velocityY : playerVelocityY,
}

// Ball
let ballWidth = 25;
let ballHeight = 25;
let ball = {
  x : boardWidth / 2,
  y : boardHeight / 2,
  width : ballWidth,
  height : ballHeight,
  velocityX : 1,
  velocityY : 2,
}

let player1Score = 0;
let player2Score = 0;

window.onload = function() {
  board = document.getElementById('board');
  board.height = boardHeight;
  board.width = boardWidth;
  context = board.getContext('2d'); // Used for drawing

  // Draw initial player1
  context.fillStyle = '#FBDB65';
  context.fillRect(player1.x, player1.y, player1.width, player1.height);

  requestAnimationFrame(update);
  document.addEventListener('keyup', movePlayer);
}

function update() {
  requestAnimationFrame(update);
  context.clearRect(0, 0, board.width, board.height);

  // Player1
  context.fillStyle = '#FBDB65';
  // player1.y += player1.velocityY;
  let nextPlayer1Y = player1.y + player1.velocityY;
  if (!outOfBounds(nextPlayer1Y)) {
    player1.y = nextPlayer1Y;
  }

  context.fillRect(player1.x, player1.y, player1.width, player1.height);

  // Player2
  // player2.y += player2.velocityY;
  let nextPlayer2Y = player2.y + player2.velocityY;
  if (!outOfBounds(nextPlayer2Y)) {
    player2.y = nextPlayer2Y;
  }
  context.fillRect(player2.x, player2.y, player2.width, player2.height);

  // // Ball
  // context.fillStyle = '#880808';
  // ball.x += ball.velocityX;
  // ball.y += ball.velocityY;
  // context.fillRect(ball.x, ball.y, ball.width, ball.height);

  // If ball touches bottom or top
  if (ball.y <= 0 || ball.y + ball.height >= boardHeight) {
    ball.velocityY *= -1; // Reverse direction on Y
  }

  if (detectCollision(ball, player1)) {
    if (ball.x <= player1.x + player1.width) {
      // Left side of ball touches right side of Player1
      ball.velocityX *= -1;
    }
  }
  else if (detectCollision(ball, player2)) {
    if (ball.x + ballWidth >= player2.x) {
      // Right side of ball touches left side of Player2
      ball.velocityX *= -1;
    }
  }

  // Game Over
  if (ball.x < 0) {
    player2Score++;
    resetGame(1);
  }
  else if (ball.x + ballWidth > boardWidth) {
    player1Score++;
    resetGame(-1);
  }

  // Score
  // context.font = '180px JetBrains Mono';
  // context.fillText(player1Score, boardWidth / 5, boardHeight / 2);
  // context.fillText(player2Score, boardWidth * (4/5) - 45, boardHeight / 2);
  
  context.fillStyle = '#404040';
  context.font = 'bold 200px JetBrains Mono';

  // Misura la larghezza e l'altezza del testo player1Score
  var player1ScoreWidth = context.measureText(player1Score).width;
  var player1ScoreHeight = 180; // Altezza del testo

  // Calcola la posizione x per centrare il testo player1Score
  var player1ScoreX = (boardWidth / 4) - (player1ScoreWidth / 2);

  // Calcola la posizione y per centrare il testo player1Score
  var player1ScoreY = (boardHeight / 2) + (player1ScoreHeight / 4); // Altezza divisa per 4 per centrare verticalmente

  // Disegna il testo player1Score centrato
  context.fillText(player1Score, player1ScoreX, player1ScoreY);

  // Misura la larghezza e l'altezza del testo player2Score
  var player2ScoreWidth = context.measureText(player2Score).width;
  var player2ScoreHeight = 180; // Altezza del testo

  // Calcola la posizione x per centrare il testo player2Score
  var player2ScoreX = (boardWidth * (3/4)) - (player2ScoreWidth / 2);

  // Calcola la posizione y per centrare il testo player2Score
  var player2ScoreY = (boardHeight / 2) + (player2ScoreHeight / 4); // Altezza divisa per 4 per centrare verticalmente

  // Disegna il testo player2Score centrato
  context.fillText(player2Score, player2ScoreX, player2ScoreY);


  // Ball (here to be OVER the scores!)
  context.fillStyle = '#880808';
  ball.x += ball.velocityX;
  ball.y += ball.velocityY;
  context.fillRect(ball.x, ball.y, ball.width, ball.height);


  context.fillStyle = '#404040';
  for (let i = 10; i < board.height; i += 25) {
    context.fillRect(board.width / 2 - 10, i, 5, 5);
  }
}

function outOfBounds(yPosition) {
  return (yPosition < 0 || yPosition + playerHeight > boardHeight);
}

function movePlayer(e) {
    // Player1
  if (e.code == 'KeyW') {
    player1.velocityY = -3;
  }
  else if (e.code == 'KeyS') {
    player1.velocityY = 3;
  }

  // Player 2
  if (e.code == 'ArrowUp') {
    player2.velocityY = -3;
  }
  else if (e.code == 'ArrowDown') {
    player2.velocityY = 3;
  }
}

function detectCollision(a, b) {
  return a.x < b.x + b.width &&
    a.x + a.width > b.x &&
    a.y < b.y + b.height &&
    a.y + a.height > b.y;
}

function resetGame(direction) {
  ball = {
  x : boardWidth / 2,
  y : boardHeight / 2,
  width : ballWidth,
  height : ballHeight,
  velocityX : direction,
  velocityY : 2,
  }
}