
#ifndef __PROGRESS_BAR_HPP_
#define __PROGRESS_BAR_HPP_

#include <iostream>
#include <string>

void progress_bar(unsigned int current, unsigned int total, std::string label) {
    std::string bar;
    for(unsigned int i = 0; i < 30; i++) {
        if(i < label.length()) bar += label[i];
        else bar += " ";
    }
    bar += "[";

    unsigned int bar_width = 40;
    unsigned int pos = (int)(bar_width * current / total);
    for(unsigned int i = 0; i < bar_width; i++) {
        if(i < pos) bar += "|";
        else bar += " ";
    }
    bar += "] (" + std::to_string((int)(100 * current / total)) + "%)\r";
    std::cout << bar;
    if(current == total) std::cout << "\n";
    else std::cout.flush();
}

#endif
