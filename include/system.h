#ifndef SYSTEM_H
#define SYSTEM_H

#include <stdio.h>

#include "utils.h"
#include "Frame.h"
#include "Map.h"
#include "Mapviewer.h"
#include "optimizer.hpp"
#include "PoseOpt.h"


void SLAMsystem(std::string commonPath, std::string yamlPath);


#endif