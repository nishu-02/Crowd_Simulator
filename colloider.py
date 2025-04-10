import pygame
import numpy as np
import random

# Window dimensions
WIDTH, HEIGHT = 1200, 800

# Colors
WHITE = (255, 255, 255)
RED = (255, 0, 0)
BLUE = (0, 0, 255)
YELLOW = (255, 255, 0)

class Particle:
    def __init__(self, x, y, vx, vy, color):
        self.x = x
        self.y = y
        self.vx = vx
        self.vy = vy
        self.color = color
        self.radius = 5

    def update(self):
        self.x += self.vx
        self.y += self.vy

        # Boundary collision
        if self.x < 0 or self.x > WIDTH:
            self.vx *= -1
        if self.y < 0 or self.y > HEIGHT:
            self.vy *= -1

    def draw(self, screen):
        pygame.draw.circle(screen, self.color, (int(self.x), int(self.y)), self.radius)

def draw_tube(screen):
    pygame.draw.circle(screen, (0, 0, 0), (WIDTH // 2, HEIGHT // 2), 300, 10)

def main():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    clock = pygame.time.Clock()

    particles = [
        Particle(WIDTH // 2 + 250, HEIGHT // 2, -5, 0, RED),
        Particle(WIDTH // 2 - 250, HEIGHT // 2, 5, 0, BLUE),
        Particle(WIDTH // 2 + 200, HEIGHT // 2 + 100, -3, -3, RED),
        Particle(WIDTH // 2 - 200, HEIGHT // 2 - 100, 3, 3, BLUE),
        Particle(WIDTH // 2 + 150, HEIGHT // 2 - 150, -2, 2, RED),
        Particle(WIDTH // 2 - 150, HEIGHT // 2 + 150, 2, -2, BLUE),
    ]

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        screen.fill(WHITE)
        draw_tube(screen)

        for particle in particles:
            particle.update()
            particle.draw(screen)

            # Collision detection
            for other in particles:
                if particle != other:
                    dx = particle.x - other.x
                    dy = particle.y - other.y
                    distance = np.sqrt(dx**2 + dy**2)
                    if distance < particle.radius + other.radius:
                        # Simulate collision by changing velocities and adding visual effects
                        particle.vx, other.vx = other.vx, particle.vx
                        particle.vy, other.vy = other.vy, particle.vy
                        
                        # Visual effect: Draw a burst of particles
                        for _ in range(10):
                            angle = random.uniform(0, 2*np.pi)
                            speed = random.uniform(2, 5)
                            vx = speed * np.cos(angle)
                            vy = speed * np.sin(angle)
                            pygame.draw.circle(screen, YELLOW, (int(particle.x), int(particle.y)), 2)

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()

if __name__ == "__main__":
    main()
