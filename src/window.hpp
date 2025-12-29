#pragma once
#include <SDL2/SDL.h>
#include <iostream>
#include <vector>

struct SDLHelper {
    SDL_Window *window = nullptr;
    SDL_Renderer *renderer = nullptr;
    SDL_Texture *texture = nullptr;
    int width = 0;
    int height = 0;
    std::vector<unsigned char> pixels;

    bool create(int w, int h, const char *title = "SDL Window") {
        width = w;
        height = h;
        pixels.resize(width * height * 3, 0);

        if (SDL_Init(SDL_INIT_VIDEO) != 0) {
            std::cerr << "SDL_Init Error: " << SDL_GetError() << "\n";
            return false;
        }

        window = SDL_CreateWindow(title, SDL_WINDOWPOS_CENTERED,
                                  SDL_WINDOWPOS_CENTERED, width, height,
                                  SDL_WINDOW_SHOWN);
        if (!window) {
            std::cerr << "SDL_CreateWindow Error: " << SDL_GetError() << "\n";
            return false;
        }

        renderer = SDL_CreateRenderer(window, -1, SDL_RENDERER_ACCELERATED);
        if (!renderer) {
            std::cerr << "SDL_CreateRenderer Error: " << SDL_GetError() << "\n";
            return false;
        }

        texture = SDL_CreateTexture(renderer, SDL_PIXELFORMAT_RGB24,
                                    SDL_TEXTUREACCESS_STREAMING, width, height);
        if (!texture) {
            std::cerr << "SDL_CreateTexture Error: " << SDL_GetError() << "\n";
            return false;
        }

        return true;
    }

    void write_pixel(int x, int y, unsigned char r, unsigned char g,
                     unsigned char b) {
        if (x < 0 || x >= width || y < 0 || y >= height)
            return;
        int idx = 3 * (y * width + x);
        pixels[idx + 0] = r;
        pixels[idx + 1] = g;
        pixels[idx + 2] = b;
    }

    void update() {
        SDL_UpdateTexture(texture, nullptr, pixels.data(), width * 3);
        SDL_RenderClear(renderer);
        SDL_RenderCopy(renderer, texture, nullptr, nullptr);
        SDL_RenderPresent(renderer);
    }

    void destroy() {
        if (texture)
            SDL_DestroyTexture(texture);
        if (renderer)
            SDL_DestroyRenderer(renderer);
        if (window)
            SDL_DestroyWindow(window);
        SDL_Quit();

        texture = nullptr;
        renderer = nullptr;
        window = nullptr;
    }
};
