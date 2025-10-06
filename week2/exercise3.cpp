#include <iostream>

enum RpsMove {
    ROCK,
    PAPER,
    SCISSORS
};

enum RpsResult {
    PLAYER1 = 1,
    DRAW = 0,
    PLAYER2 = -1
};

RpsResult simulateRps(RpsMove player1, RpsMove player2) {
    int comboIndex = player1 + 3 * player2;
    switch (comboIndex) {
        case 1:
        case 5:
        case 6:
            return PLAYER1;
        case 2:
        case 3:
        case 7:
            return PLAYER2;
        default:
            return DRAW;
    }
}

int main() {
    std::cout << simulateRps(ROCK, PAPER) << std::endl;
    std::cout << simulateRps(PAPER, ROCK) << std::endl;
    std::cout << simulateRps(SCISSORS, SCISSORS) << std::endl;
    std::cout << simulateRps(SCISSORS, PAPER) << std::endl;
    std::cout << simulateRps(SCISSORS, ROCK) << std::endl;
    std::cout << simulateRps(PAPER, PAPER) << std::endl;
}