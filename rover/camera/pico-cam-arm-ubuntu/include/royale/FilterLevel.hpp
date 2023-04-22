/****************************************************************************\
 * Copyright (C) 2021 pmdtechnologies ag
 *
 * THIS CODE AND INFORMATION ARE PROVIDED "AS IS" WITHOUT WARRANTY OF ANY
 * KIND, EITHER EXPRESSED OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND/OR FITNESS FOR A
 * PARTICULAR PURPOSE.
 *
 \****************************************************************************/

#pragma once

#include <royale/FilterPreset.hpp>

namespace royale
{
    /** @deprecated */
    typedef FilterPreset FilterLevel;

    /** @deprecated */
    ROYALE_API royale::String getFilterLevelName (royale::FilterLevel level);

    /** @deprecated */
    ROYALE_API royale::FilterLevel getFilterLevelFromName (royale::String name);
}
