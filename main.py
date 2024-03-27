import matplotlib.pyplot as plt
import numpy as np




class DrawSin: #클래스 생성
    def __init__(self, t, A, f, b): # 생성자
        self.t = t
        self.A = A
        self.f = f
        self.b = b

    def draw_sin(self, t, A, f, b):
        # 함수를 만들기
        y = A * np.sin(2 * np.pi * f * t) + b
        # 사인의 파형을 만든다
        # print(y)

        plt.figure(figsize=(12, 6))
        # matplotlib에 출력할 사이즈를 설정한다
        plt.plot(t, y)
        # x, y 값을 넣어 둔다
        plt.grid()
        plt.show()

s=DrawSin(2, 2,5,0)

t = np.arange(0, 2, 0.001)
#0~6까지 0.01씩 넘 파이의 배열이 만들어 진다
# print(t)
draw_sin(t, 2,5,0)
