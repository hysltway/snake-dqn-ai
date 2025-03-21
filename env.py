import random
import pygame
import sys
import numpy as np
from PIL import Image
from torchvision import transforms


class Snake:
    def __init__(self):
        self.snake_speed = 100  # 贪吃蛇的速度
        self.windows_width = 600
        self.windows_height = 600  # 游戏窗口的大小
        self.cell_size = 50
        self.map_width = int(self.windows_width / self.cell_size)
        self.map_height = int(self.windows_height / self.cell_size)
        self.white = (255, 255, 255)
        self.black = (0, 0, 0)
        self.gray = (230, 230, 230)
        self.dark_gray = (40, 40, 40)
        self.DARKGreen = (0, 155, 0)
        self.Green = (0, 255, 0)
        self.Red = (255, 0, 0)
        self.blue = (0, 0, 255)
        self.dark_blue = (0, 0, 139)
        self.BG_COLOR = self.white

        self.UP = 0
        self.DOWN = 1
        self.LEFT = 2
        self.RIGHT = 3

        self.HEAD = 0

        pygame.init()
        self.snake_speed_clock = pygame.time.Clock()

        [self.snake_coords, self.direction, self.food, self.state] = [None, None, None, None]

    def reset(self):
        startx = random.randint(3, self.map_width - 8)
        starty = random.randint(3, self.map_height - 8)
        self.snake_coords = [{'x': startx, 'y': starty},
                        {'x': startx - 1, 'y': starty},
                        {'x': startx - 2, 'y': starty}]
        self.direction = self.RIGHT
        self.food = self.get_random_location()
        return self.getState()

    def step(self, action):
        if action == self.LEFT and self.direction != self.RIGHT:
            self.direction = self.LEFT
        elif action == self.RIGHT and self.direction != self.LEFT:
            self.direction = self.RIGHT
        elif action == self.UP and self.direction != self.DOWN:
            self.direction = self.UP
        elif action == self.DOWN and self.direction != self.UP:
            self.direction = self.DOWN
        self.move_snake(self.direction, self.snake_coords)
        ret = self.snake_is_alive(self.snake_coords)
        d = True if not ret else False
        flag = self.snake_is_eat_food(self.snake_coords, self.food)
        reward = self.getReward(flag, d)

        return [self.getState(), reward, d, None]

    def getReward(self, flag, d):
        reward = 0
        if flag:
            reward += 2
        if d:
            reward -= 0.5
        return reward

    def render(self):
        self.screen = pygame.display.set_mode((self.windows_width, self.windows_height))
        self.screen.fill(self.BG_COLOR)
        self.draw_snake(self.screen,self.snake_coords)
        self.draw_food(self.screen,self.food)
        self.draw_score(self.screen,len(self.snake_coords)-3)
        pygame.display.update()
        self.snake_speed_clock.tick(self.snake_speed)

    def getState(self):
        # 基础部分 6个维度
        [xhead, yhead] = [self.snake_coords[self.HEAD]['x'], self.snake_coords[self.HEAD]['y']]
        [xfood, yfood] = [self.food['x'], self.food['y']]
        deltax = (xfood - xhead) / self.map_width
        deltay = (yfood - yhead) / self.map_height
        checkPoint = [[xhead, yhead - 1], [xhead - 1, yhead], [xhead, yhead + 1], [xhead + 1, yhead]]
        tem = [0, 0, 0, 0]
        for coord in self.snake_coords[1:]:
            if [coord['x'], coord['y']] in checkPoint:
                index = checkPoint.index([coord['x'], coord['y']])
                tem[index] = 1
        for i, point in enumerate(checkPoint):
            if point[0] >= self.map_width or point[0] < 0 or point[1] >= self.map_height or point[1] < 0:
                tem[i] = 1
        state = [deltax, deltay]
        state.extend(tem)

        # 加入蛇身体中部和尾部位置信息  增加4个维度
        length = len(self.snake_coords)
        snake_mid = [self.snake_coords[int(length / 2)]['x'] - xhead, self.snake_coords[int(length / 2)]['y'] - yhead]
        snake_tail = [self.snake_coords[-1]['x'] - xhead, self.snake_coords[-1]['y'] - yhead]
        state.extend(snake_mid+snake_tail)
        return state

    def draw_food(self, screen, food):
        x = food['x'] * self.cell_size
        y = food['y'] * self.cell_size
        appleRect = pygame.Rect(x, y, self.cell_size, self.cell_size)
        pygame.draw.rect(screen, self.Red, appleRect)

    # 将贪吃蛇画出来
    def draw_snake(self, screen, snake_coords):
        for i, coord in enumerate(snake_coords):
            color = self.Green if i == 0 else self.dark_blue
            x = coord['x'] * self.cell_size
            y = coord['y'] * self.cell_size
            wormSegmentRect = pygame.Rect(x, y, self.cell_size, self.cell_size)
            pygame.draw.rect(screen, self.dark_blue, wormSegmentRect)
            wormInnerSegmentRect = pygame.Rect(  # 蛇身子里面的第二层亮绿色
                x + 4, y + 4, self.cell_size - 8, self.cell_size - 8)
            pygame.draw.rect(screen, color, wormInnerSegmentRect)

    # 移动贪吃蛇
    def move_snake(self,direction, snake_coords):
        if direction == self.UP:
            newHead = {'x': snake_coords[self.HEAD]['x'], 'y': snake_coords[self.HEAD]['y'] - 1}
        elif direction == self.DOWN:
            newHead = {'x': snake_coords[self.HEAD]['x'], 'y': snake_coords[self.HEAD]['y'] + 1}
        elif direction == self.LEFT:
            newHead = {'x': snake_coords[self.HEAD]['x'] - 1, 'y': snake_coords[self.HEAD]['y']}
        elif direction == self.RIGHT:
            newHead = {'x': snake_coords[self.HEAD]['x'] + 1, 'y': snake_coords[self.HEAD]['y']}
        else:
            newHead = None
            raise Exception('error for direction!')

        snake_coords.insert(0, newHead)

    def snake_is_alive(self,snake_coords):
        tag = True
        if snake_coords[self.HEAD]['x'] == -1 or snake_coords[self.HEAD]['x'] == self.map_width or snake_coords[self.HEAD]['y'] == -1 or \
                snake_coords[self.HEAD]['y'] == self.map_height:
            tag = False
        for snake_body in snake_coords[1:]:
            if snake_body['x'] == snake_coords[self.HEAD]['x'] and snake_body['y'] == snake_coords[self.HEAD]['y']:
                tag = False
        return tag

    def snake_is_eat_food(self, snake_coords, food):
        flag = False
        if snake_coords[self.HEAD]['x'] == food['x'] and snake_coords[self.HEAD]['y'] == food['y']:
            while True:
                food['x'] = random.randint(0, self.map_width - 1)
                food['y'] = random.randint(0, self.map_height - 1)
                tag = 0
                for coord in snake_coords:
                    if [coord['x'], coord['y']] == [food['x'], food['y']]:
                        tag = 1
                        break
                if tag == 1:
                    continue
                break
            flag = True
        else:
            del snake_coords[-1]
        return flag

    def get_random_location(self):
        return {'x': random.randint(0, self.map_width - 1), 'y': random.randint(0, self.map_height - 1)}

    def draw_score(self,screen, score):
        font = pygame.font.Font('myfont.ttf', 30)
        scoreSurf = font.render('得分: %s' % score, True, self.black)
        scoreRect = scoreSurf.get_rect()
        scoreRect.topleft = (self.windows_width - 120, 10)
        screen.blit(scoreSurf, scoreRect)

    @staticmethod
    # 程序终止
    def terminate():
        pygame.quit()
        sys.exit()

    def screenTensor(self):
        self.render()
        image_data = pygame.surfarray.array3d(pygame.display.get_surface()).transpose((1, 0, 2))
        img = Image.fromarray(image_data.astype(np.uint8))
        transform = transforms.Compose([
                                        transforms.Resize((50, 50)),
                                        transforms.ToTensor(),
                                        ])
        img_tensor = transform(img)

        return img_tensor

