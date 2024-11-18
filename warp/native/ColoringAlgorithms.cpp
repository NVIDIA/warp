#include "ColoringAlgorithms.h"

#include "../Json/json.hpp"

#include <array>
#include <iostream>

using std::cout;
using std::endl;

GAIA::GraphColoring::GraphColor::GraphColor(const Graph& inGraph)
{
    graph.resize(inGraph.numNodes);

    for (size_t i = 0; i < inGraph.edges.size(); i++)
    {
        const std::array<int, 2>& e = inGraph.edges[i];
        graph[e[0]].push_back(e[1]);
        graph[e[1]].push_back(e[0]);
    }

    for (size_t i = 0; i < graph.size(); i++)
    {
        graph_colors.push_back(-1);
    }
}

GAIA::GraphColoring::GraphColor::GraphColor(const std::vector<vector<int>>& inGraph)
{
    graph = inGraph;
    for (size_t i = 0; i < graph.size(); i++)
    {
        graph_colors.push_back(-1);
    }
}

int GAIA::GraphColoring::GraphColor::get_num_colors()
{
    int numColors = 0;
    for (size_t i = 0; i < graph.size(); i++)
    {
        if (graph_colors[i] + 1 > numColors)
        {
            numColors = graph_colors[i] + 1;
        }
    };
    return numColors;
}

bool GAIA::GraphColoring::GraphColor::is_valid()
{
    if (this->graph_colors.size() == 0 || this->graph.size() != this->graph_colors.size()) {
        return false;
    }
    for (size_t i = 0; i < graph.size(); i++) {
        if (graph_colors[i] == -1) {
            return false;
        }
        for (size_t j = 0; j < graph[i].size(); j++) {
            int neiNode = graph[i][j];
            if (graph_colors[i] == graph_colors[neiNode]) {
                return false;
            }
        }
    }
    return true;
}

void GAIA::GraphColoring::GraphColor::convertToColoredCategories()
{
    categories.clear();
    size_t numColors = get_num_colors();
    categories.resize(numColors);

    for (int iV = 0; iV < size(); iV++) {
        int color = get_color(iV);
        // in this 
        categories[color].push_back(iV);
    }
}

float GAIA::GraphColoring::GraphColor::findLargestSmallestCategories(int& biggestCategory, int& smallestCategory)
{
    if (categories.size() == 0)
    {
        biggestCategory = -1;
        smallestCategory = -1;

        return 1;
    }

    size_t maxSize = categories[0].size();
    biggestCategory = 0;
    size_t minSize = categories[0].size();
    smallestCategory = 0;

    for (size_t iColor = 0; iColor < categories.size(); iColor++)
    {
        if (maxSize < categories[iColor].size()) {
            biggestCategory = iColor;
            maxSize = categories[iColor].size();
        }

        if (minSize > categories[iColor].size())
        {
            smallestCategory = iColor;
            minSize = categories[iColor].size();
        }
    }

    return float(categories[biggestCategory].size()) / float(categories[smallestCategory].size());
}

int GAIA::GraphColoring::GraphColor::findChangableNodeInCategory(int sourceColor, int destinationColor)
{
    auto& sourceCategory = categories[sourceColor];
    for (size_t iNode = 0; iNode < sourceCategory.size(); iNode++)
    {
        if (changable(sourceCategory[iNode], destinationColor)) {
            return iNode;
        }
    }
    return -1;
}

void GAIA::GraphColoring::GraphColor::changeColor(int sourceColor, int categoryId, int destinationColor)
{
    int nodeId = categories[sourceColor][categoryId];
    graph_colors[nodeId] = destinationColor;

    if (categories.size())
    {
        categories[sourceColor].erase(categories[sourceColor].begin() + categoryId);
        categories[destinationColor].push_back(nodeId);
    }
}

bool GAIA::GraphColoring::GraphColor::changable(int node, int destinationColor)
{
    // loop through node and see if it has destinationColor
    for (size_t i = 0; i < graph[node].size(); i++)
    {
        int neiId = graph[node][i];
        if (graph_colors[neiId] == destinationColor)
        {
            return false;
        }
    }
    return true;
}

void GAIA::GraphColoring::GraphColor::balanceColoredCategories(float goalMaxMinRatio)
{
    float maxMinRatio = -1.f;

    do
    {
        int biggestCategory = -1, smallestCategory = -1;

        maxMinRatio = findLargestSmallestCategories(biggestCategory, smallestCategory);

        // find a availiable vertex from the biggest category to move to the smallest category
        int changableId = findChangableNodeInCategory(biggestCategory, smallestCategory);
        if (changableId == -1)
        {
            for (size_t iColor = 0; iColor < categories.size(); iColor++)
            {
                if (iColor == biggestCategory || iColor == smallestCategory)
                {
                    continue;
                }

                changableId = findChangableNodeInCategory(iColor, smallestCategory);

                if (changableId != -1)
                {
                    biggestCategory = iColor;

                    break;
                }
            }
        }


        if (changableId == -1)
        {
            std::cout << "The graph is not opimizable anymore, terminated with a max/min ratio: " << maxMinRatio << std::endl;
            return;
        }
        changeColor(biggestCategory, changableId, smallestCategory);

        // change the color of changable id

    } while (maxMinRatio > goalMaxMinRatio);

    std::cout << "The graph optimization terminated with a max/min ratio: " << maxMinRatio << std::endl;

}

void GAIA::GraphColoring::GraphColor::saveColoringCategories(std::string outputFile)
{
    for (size_t iColor = 0; iColor < this->get_num_colors(); iColor++)
    {
        cout << "num nodes in colored category " << iColor << ": " << categories[iColor].size() << std::endl;
    }

    nlohmann::json j = categories;

    std::ofstream ofs(outputFile);
    ofs << j.dump(-1);
}
