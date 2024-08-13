public class Particle {
  float x, y, angle;
  float speed = 5;
  int c;

  Particle(float x, float y, float angle, int c) {
    this.x = x;
    this.y = y;
    this.angle = angle;
    this.c = c;
  }

  // Update the particle position
  void update() {
    this.x += Math.cos(this.angle) * this.speed;
    this.y += Math.sin(this.angle) * this.speed;
  }

  void show() {

  }
}

