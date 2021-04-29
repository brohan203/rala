/*!
 * @file pile.cpp
 *
 * @brief Pile class source file
 */

#include <algorithm>
#include <sstream>
#include <deque>

#include "overlap.hpp"
#include "pile.hpp"
#include "functions.cuh"

namespace rala {

// Subpile = doube ended queue
using Subpile = std::deque<std::pair<int32_t, int32_t>>;

// Remove back elements until value (second in pair) of last element < value of new pair 
void subpileAdd(Subpile& src, int32_t value, int32_t position) {
    while (!src.empty() && src.back().second <= value) {
        src.pop_back();
    }
    src.emplace_back(position, value);
}

// Remove front elements until given position
void subpileUpdate(Subpile& src, int32_t position) {
    while (!src.empty() && src.front().first <= position) {
        src.pop_front();
    }
}

void intervalMerge(std::vector<std::pair<uint32_t, uint32_t>>& intervals) {

    std::vector<std::pair<uint32_t, uint32_t>> tmp;
    std::vector<bool> is_merged(intervals.size(), false);
    for (uint32_t i = 0; i < intervals.size(); ++i) {
        if (is_merged[i]) {
            continue;
        }
        for (uint32_t j = 0; j < intervals.size(); ++j) {
            if (i != j && !is_merged[j] &&
                intervals[i].first < intervals[j].second &&
                intervals[i].second > intervals[j].first) {

                is_merged[j] = true;
                intervals[i].first = std::min(intervals[i].first, intervals[j].first);
                intervals[i].second = std::max(intervals[i].second, intervals[j].second);
            }
        }
        tmp.emplace_back(intervals[i].first, intervals[i].second);
    }
    intervals.swap(tmp);
}

std::unique_ptr<Pile> createPile(uint64_t id, uint32_t read_length) {
    return std::unique_ptr<Pile>(new Pile(id, read_length));
}

std::unique_ptr<Pile> createRoPile(uint32_t begin, uint64_t id, uint32_t read_length) {
    return std::unique_ptr<Pile>(new Pile(begin, id, read_length));
}

Pile::Pile(uint64_t id, uint32_t read_length)
        : id_(id), begin_(0), end_(read_length), p10_(0), median_(0),
        data_(end_ - begin_, 0), repeat_hills_(), repeat_hill_coverage_(),
        chimeric_pits_(), chimeric_hills_(), chimeric_hill_coverage_() {
}

Pile::Pile(uint32_t begin, uint64_t id, uint32_t read_length)
        : id_(id), begin_(0), end_( (read_length) ), seq_begin_(begin),
        seq_end_(begin+read_length), p10_(0), median_(0), data_(end_ - begin_, 0),
        repeat_hills_(), repeat_hill_coverage_(), chimeric_pits_(), 
        chimeric_hills_(), chimeric_hill_coverage_() {
}

// =============================================================================================
// Upper bound, basically copied from C++ standard library source code
// Inputs: overlaps (begins or ends), first and last (may remove this later), location we're looking for
uint32_t upper_bound(std::vector<uint32_t> &overlaps, uint32_t first, uint32_t last, const uint32_t location) {
    // Initialize
    // Middle is the middle of first and last
    // Length is distance between first and last
    uint32_t middle, len, step;
    len = last - first;

    while(len > 0) {
        middle = first;
        step = len / 2;
        middle = middle + step;
        if(!(location < overlaps[middle])) {
            first = ++middle;
            len -= step + 1;
        }
        else {
            len = step;
        }
    }
    return first - 1;
}

uint32_t Pile::find_histo_height(uint32_t location, std::vector<uint32_t> &overlap_begins, std::vector<uint32_t> &overlap_ends) {
    
    // Adjust the start position within this pile (location) to the global position
    const uint32_t thisLocation = location + seq_begin_;

    return ( upper_bound(overlap_begins, 0, overlap_begins.size()-1, thisLocation) - 
             upper_bound(overlap_ends, 0, overlap_ends.size()-1, thisLocation    ) );
}

// =========================================================================================

// NOTES:
// We have left and right subpiles. Subpiles are just sliding windows.
// The two subpiles are nearly adjacent
    // left spans from (i - k) to (i - 1)
    // right spans from (i + 1) to (i + k)
    // i is between them but contained in neither
// Always check i vs max( (i+1) to (i+k) ) and max( (i-k) to (i-1))
std::vector<std::pair<uint32_t, uint32_t>> Pile::brute_find_slopes(double q, std::vector<uint32_t> &overlap_begins, std::vector<uint32_t> &overlap_ends) {

    std::vector<std::pair<uint32_t, uint32_t>> slope_regions;

    int32_t k = 847;
    int32_t read_length = end_ - begin_;

    // Trying to replace slope_regions with an array
    uint32_t slope_region_begins[k];
    memset( slope_region_begins, 0, k*sizeof(uint32_t));
    uint32_t slope_region_ends[k];
    memset( slope_region_begins, 0, k*sizeof(uint32_t));
    uint32_t slope_regions_counter = 0;

    // Turned subpiles into arrays
    // Down corresponds to left subpile
    uint32_t left_subpile[k];
    memset( left_subpile, 0, k*sizeof(uint32_t));
    uint32_t first_down = 0, last_down = 0;
    bool found_down = false;

    // Up corresponds to right subpile
    uint32_t right_subpile[k];
    memset( right_subpile, 0, k*sizeof(uint32_t));
    uint32_t first_up = 0, last_up = 0;
    bool found_up = false;

    // Initialize right subpile with first k elements
    for (int32_t i = 0; i < k; ++i) {
        right_subpile[i] = find_histo_height(i, overlap_begins, overlap_ends);
    }

    // I believe this will eventually be parallelized
    for (int32_t i = 0; i < read_length; ++i) {

        // Last position in the array
        uint32_t subpile_end   = i % k;
        // First position in the array
        uint32_t subpile_begin = (i+1) % k;

        // Update left and right subpiles
        // This keeps left subpile from (i - k - 1) to (i - 1)
        if (i > 0) {
            // Old line = subpileAdd(left_subpile, data_[i - 1], i - 1);
            left_subpile[subpile_end] = find_histo_height(i - 1, overlap_begins, overlap_ends);
        }
        // Last element of left subpile is i+k
        // This keeps right subpile from (i + 1) to (i + k)
        if (i < read_length - k) {
            // Old line = subpileAdd(right_subpile, data_[i + k], i + k);
            right_subpile[subpile_end] = find_histo_height(i + k, overlap_begins, overlap_ends);
        }

        uint32_t left_max  = 0;
        uint32_t right_max = 0;
        for(int x = 0; x < k; ++x) {
            if(left_subpile[x] > left_max) {
                left_max = left_subpile[x];
            }
            if(right_subpile[x] > right_max) {
                right_max = right_subpile[x];
            }
        }


        // Old line = int32_t current_value = data_[i] * q;
        int32_t current_value = find_histo_height(i, overlap_begins, overlap_ends) * q;

        // If first element of left subpile > (current_value * q)
        if (i != 0 && left_subpile[subpile_begin] > current_value) {
            if (found_down) {
                if (i - last_down > 1) {
                    slope_regions.emplace_back(first_down, last_down);
                    slope_region_begins[slope_regions_counter] = first_down;
                    slope_region_ends[slope_regions_counter]   = last_down;
                    ++slope_regions_counter;
                    first_down = i;
                }
            } else {
                found_down = true;
                first_down = i;
            }
            last_down = i;
        }
        if (i != (read_length - 1) && right_subpile[subpile_begin] > current_value) {
            if (found_up) {
                if (i - last_up > 1) {
                    slope_regions.emplace_back(first_up, last_up);
                    slope_region_begins[slope_regions_counter] = first_up;
                    slope_region_ends[slope_regions_counter]   = last_up;
                    ++slope_regions_counter;
                    first_up = i;
                }
            } else {
                found_up = true;
                first_up = i;
            }
            last_up = i;
        }
    }
    if (found_down) {
        slope_regions.emplace_back(first_down, last_down);
        slope_region_begins[slope_regions_counter] = first_down;
        slope_region_ends[slope_regions_counter]   = last_down;
        ++slope_regions_counter;
    }
    if (found_up) {
        slope_regions.emplace_back(first_up, last_up);
        slope_region_begins[slope_regions_counter] = first_up;
        slope_region_ends[slope_regions_counter]   = last_up;
        ++slope_regions_counter;
    }

    if(slope_regions_counter < 2) {
        return slope_regions;
    }

    printf("239\n");
    uint32_t iterations = 0

    while (true) {
        // Implementation of insertion sort for our slope_region_begins/ends arrays
        // Sorts based on slope_region_begins
        int i, key_begins, key_ends, j;
        for (i = 1; i < slope_regions_counter; i++)
        {
            key_begins = slope_region_begins[i];
            key_ends = slope_region_ends[i];
            j = i - 1;
    
            /* Move elements of arr[0..i-1], that are
            greater than key, to one position ahead
            of their current position */
            while (j >= 0 && slope_region_begins[j] > key_begins)
            {
                slope_region_begins[j + 1] = slope_region_begins[j];
                slope_region_ends[j + 1] = slope_region_ends[j];
                j = j - 1;
            }
            slope_region_begins[j + 1] = key_begins;
            slope_region_ends[j + 1] = key_ends;
        }

        bool is_changed = false;
        for (uint32_t i = 0; i < slope_regions_counter; ++i) {
            if (slope_region_ends[i] < (slope_region_ends[i + 1])) {
                continue;
            }

            std::vector<std::pair<uint32_t, uint32_t>> subregions;
            uint32_t subregion_begins[k];
            memset( subregion_begins, 0, k*sizeof(uint32_t) );
            uint32_t subregion_ends[k];
            memset( subregion_ends, 0, k*sizeof(uint32_t) );
            uint32_t subregions_counter = 0;
            if (slope_region_begins[i] & 1) {
                // STUCK HERE

                // Clear right subpile
                for(int x = 0; x < k; ++x) {
                    right_subpile[x] = 0;
                }

                found_up = false;
                uint32_t subpile_begin = slope_region_begins[i];
                uint32_t subpile_end;
                if(slope_region_ends[i+1] > 0) {
                    subpile_end = std::min(slope_region_ends[i], slope_region_ends[i + 1]);
                }
                else {
                    subpile_end = slope_region_ends[i];
                }

                for (uint32_t j = subpile_begin; j < subpile_end + 1; ++j) {
                    right_subpile[j - subpile_begin] = find_histo_height(j, overlap_begins, overlap_ends);
                }
                for (uint32_t j = subpile_begin; j < subpile_end; ++j) {

                    uint32_t max = 0;
                    for(int x = 0; x < k; ++x) {
                        if(right_subpile[x] > max) {
                            max = right_subpile[x];
                        }
                    }
                    if (find_histo_height(j, overlap_begins, overlap_ends) * q < max) {
                        if (found_up) {
                            if (j - last_up > 1) {
                                subregion_begins[subregions_counter] = first_up;
                                subregion_ends[subregions_counter]   = last_up;
                                ++subregions_counter;
                                first_up = j;
                            }
                        } else {
                            found_up = true;
                            first_up = j;
                        }
                        last_up = j;
                    }
                    right_subpile[(j - subpile_begin) % k] = find_histo_height(j, overlap_begins, overlap_ends);
                }
                if (found_up) {
                    subregion_begins[subregions_counter] = first_up;
                    subregion_ends[subregions_counter]   = last_up;
                    ++subregions_counter;
                }

                for(int x = 0; x < subregions_counter; ++x) {
                    slope_regions.emplace_back(subregion_begins[x], subregion_ends[x]);
                    slope_region_begins[slope_regions_counter] = subregion_begins[x];
                    slope_region_ends[slope_regions_counter]   = subregion_ends[x];
                    ++slope_regions_counter;
                }
                slope_region_begins[i] = subpile_end << 1 | 1;

            } else {
                if (slope_region_ends[i] == (slope_region_begins[i + 1])) {
                    continue;
                }

                uint32_t else_subregion_counter = subregions_counter;

                // Clear left subpile
                for(int x = 0; x < k; ++x) {
                    left_subpile[x] = 0;
                }
                found_down = false;

                uint32_t subpile_begin = std::max(slope_region_begins[i],
                    slope_region_begins[i + 1]);
                uint32_t subpile_end = slope_region_ends[i];

                // Rebuild left subpile from this specific begin to end
                for (uint32_t j = subpile_begin; j < subpile_end + 1; ++j) {
                    left_subpile[j - subpile_begin] = find_histo_height(j, overlap_begins, overlap_ends);
                }

                for (uint32_t j = subpile_begin; j < subpile_end + 1; ++j) {
                    // Find the max of the left subpile
                    uint32_t max = 0;
                    for(int x = 0; x < k; ++x) {
                        if(left_subpile[x] > max) {
                            max = left_subpile[x];
                        }
                    }
                    if (find_histo_height(j, overlap_begins, overlap_ends) * q < max) {
                        if (found_down) {
                            if (j - last_down > 1) {
                                subregion_begins[subregions_counter] = first_down;
                                subregion_ends[subregions_counter]   = last_down;
                                ++subregions_counter;
                                first_down = j;
                            }
                        } else {
                            found_down = true;
                            first_down = j;
                        }
                        last_down = j;
                    }
                    left_subpile[(j - subpile_begin) % k] = find_histo_height(j, overlap_begins, overlap_ends);
                }
                if (found_down) {
                    subregion_begins[subregions_counter] = first_down;
                    subregion_ends[subregions_counter]   = last_down;
                    ++subregions_counter;
                }

                for(int x = else_subregion_counter; x < subregions_counter; ++x) {
                    slope_regions.emplace_back(subregion_begins[x], subregion_ends[x]);
                    slope_region_begins[slope_regions_counter] = subregion_begins[x];
                    slope_region_ends[slope_regions_counter]   = subregion_ends[x];
                    ++slope_regions_counter;
                }

                slope_region_ends[i] = subpile_begin;
            }

            is_changed = true;
            break;
        }

        if (!is_changed) {
            break;
        }
    }

    printf("386 brute_slopes at a pile\n");

    // narrow slope regions
    for (uint32_t i = 0; i < slope_regions_counter - 1; ++i) {
        if ((slope_region_begins[i] & 1) && !(slope_region_begins[i + 1] & 1)) {

            uint32_t subpile_begin = slope_region_ends[i];
            uint32_t subpile_end = slope_region_begins[i + 1];

            if (subpile_end - subpile_begin > static_cast<uint32_t>(k)) {
                continue;
            }

            uint16_t max_subpile_coverage = 0;
            for (uint32_t j = subpile_begin + 1; j < subpile_end; ++j) {
                if((uint16_t)find_histo_height(j, overlap_begins, overlap_ends) > max_subpile_coverage) {
                    max_subpile_coverage = (uint16_t)find_histo_height(j, overlap_begins, overlap_ends);
                }
            }

            uint32_t last_valid_point = slope_region_begins[i];
            for (uint32_t j = slope_region_begins[i]; j <= subpile_begin; ++j) {
                if (max_subpile_coverage > find_histo_height(j, overlap_begins, overlap_ends) * q) {
                    last_valid_point = j;
                }
            }

            uint32_t first_valid_point = slope_region_ends[i + 1];
            for (uint32_t j = subpile_end; j <= slope_region_ends[i + 1]; ++j) {
                if (max_subpile_coverage > find_histo_height(j, overlap_begins, overlap_ends) * q) {
                    first_valid_point = j;
                    break;
                }
            }

            slope_region_ends[i] = last_valid_point;
            slope_region_begins[i + 1] = first_valid_point << 1 | 0;
        }
    }

    return slope_regions;
}

// =========================================================================================

std::vector<std::pair<uint32_t, uint32_t>> Pile::find_slopes(double q, std::vector<uint32_t> &overlap_begins, std::vector<uint32_t> &overlap_ends) {

    std::vector<std::pair<uint32_t, uint32_t>> slope_regions;

    int32_t k = 847;
    int32_t read_length = end_;

    // Just a double ended queue
    Subpile left_subpile;
    uint32_t first_down = 0, last_down = 0;
    bool found_down = false;

    // Just a double ended queue
    Subpile right_subpile;
    uint32_t first_up = 0, last_up = 0;
    bool found_up = false;

    // find slope regions
    for (int32_t i = 0; i < k; ++i) {
        subpileAdd(right_subpile, find_histo_height(i, overlap_begins, overlap_ends), i);
    }
    for (int32_t i = 0; i < read_length; ++i) {
        if (i > 0) {
            subpileAdd(left_subpile, find_histo_height(i - 1, overlap_begins, overlap_ends), i - 1);
        }
        subpileUpdate(left_subpile, i - 1 - k);

        if (i < read_length - k) {
            subpileAdd(right_subpile, find_histo_height(i + k, overlap_begins, overlap_ends), i + k);
        }
        subpileUpdate(right_subpile, i);

        int32_t current_value = find_histo_height(i, overlap_begins, overlap_ends) * q;
        // Set last down if above a certain threshold
        if (i != 0 && left_subpile.front().second > current_value) {
            if (found_down) {
                if (i - last_down > 1) {
                    slope_regions.emplace_back(first_down << 1 | 0, last_down);
                    first_down = i;
                }
            } else {
                found_down = true;
                first_down = i;
            }
            last_down = i;
        }
        if (i != (read_length - 1) && right_subpile.front().second > current_value) {
            if (found_up) {
                if (i - last_up > 1) {
                    slope_regions.emplace_back(first_up << 1 | 1, last_up);
                    first_up = i;
                }
            } else {
                found_up = true;
                first_up = i;
            }
            last_up = i;
        }
    }
    if (found_down) {
        slope_regions.emplace_back(first_down << 1 | 0, last_down);
    }
    if (found_up) {
        slope_regions.emplace_back(first_up << 1 | 1, last_up);
    }

    if (slope_regions.empty()) {
        return slope_regions;
    }

    if(slope_regions.size() > k) {
        printf("Slope regions larger than k: %d\n", (slope_regions.size()));
    }

    while (true) {
        std::sort(slope_regions.begin(), slope_regions.end());

        bool is_changed = false;
        for (uint32_t i = 0; i < slope_regions.size() - 1; ++i) {
            if (slope_regions[i].second < (slope_regions[i + 1].first >> 1)) {
                continue;
            }

            std::vector<std::pair<uint32_t, uint32_t>> subregions;
            // ===================================================
            if (slope_regions[i].first & 1) {
                right_subpile.clear();
                found_up = false;

                uint32_t subpile_begin = slope_regions[i].first >> 1;
                uint32_t subpile_end = std::min(slope_regions[i].second,
                    slope_regions[i + 1].second);

                for (uint32_t j = subpile_begin; j < subpile_end + 1; ++j) {
                    subpileAdd(right_subpile, find_histo_height(j, overlap_begins, overlap_ends), j);
                }
                for (uint32_t j = subpile_begin; j < subpile_end; ++j) {
                    subpileUpdate(right_subpile, j);
                    if (find_histo_height(j, overlap_begins, overlap_ends) * q < right_subpile.front().second) {
                        if (found_up) {
                            if (j - last_up > 1) {
                                subregions.emplace_back(first_up, last_up);
                                first_up = j;
                            }
                        } else {
                            found_up = true;
                            first_up = j;
                        }
                        last_up = j;
                    }
                }
                if (found_up) {
                    subregions.emplace_back(first_up, last_up);
                }

                for (const auto& it: subregions) {
                    slope_regions.emplace_back(it.first << 1 | 1, it.second);
                }
                slope_regions[i].first = subpile_end << 1 | 1;
            // ===================================================
            } else {
                if (slope_regions[i].second == (slope_regions[i + 1].first >> 1)) {
                    continue;
                }

                left_subpile.clear();
                found_down = false;

                uint32_t subpile_begin = std::max(slope_regions[i].first >> 1,
                    slope_regions[i + 1].first >> 1);
                uint32_t subpile_end = slope_regions[i].second;

                for (uint32_t j = subpile_begin; j < subpile_end + 1; ++j) {
                    if (!left_subpile.empty() && find_histo_height(j, overlap_begins, overlap_ends) * q < left_subpile.front().second) {
                        if (found_down) {
                            if (j - last_down > 1) {
                                subregions.emplace_back(first_down, last_down);
                                first_down = j;
                            }
                        } else {
                            found_down = true;
                            first_down = j;
                        }
                        last_down = j;
                    }
                    subpileAdd(left_subpile, find_histo_height(j, overlap_begins, overlap_ends), j);
                }
                if (found_down) {
                    subregions.emplace_back(first_down, last_down);
                }

                for (const auto& it: subregions) {
                    slope_regions.emplace_back(it.first << 1 | 0, it.second);
                }
                slope_regions[i].second = subpile_begin;
            }
            // ===================================================

            is_changed = true;
            break;
        }

        if (!is_changed) {
            break;
        }
    }

    // narrow slope regions
    for (uint32_t i = 0; i < slope_regions.size() - 1; ++i) {
        if ((slope_regions[i].first & 1) && !(slope_regions[i + 1].first & 1)) {

            uint32_t subpile_begin = slope_regions[i].second;
            uint32_t subpile_end = slope_regions[i + 1].first >> 1;

            if (subpile_end - subpile_begin > static_cast<uint32_t>(k)) {
                continue;
            }

            uint16_t max_subpile_coverage = 0;
            for (uint32_t j = subpile_begin + 1; j < subpile_end; ++j) {
                max_subpile_coverage = std::max(max_subpile_coverage, (uint16_t)find_histo_height(j, overlap_begins, overlap_ends));
            }

            uint32_t last_valid_point = slope_regions[i].first >> 1;
            for (uint32_t j = slope_regions[i].first >> 1; j <= subpile_begin; ++j) {
                if (max_subpile_coverage > find_histo_height(j, overlap_begins, overlap_ends) * q) {
                    last_valid_point = j;
                }
            }

            uint32_t first_valid_point = slope_regions[i + 1].second;
            for (uint32_t j = subpile_end; j <= slope_regions[i + 1].second; ++j) {
                if (max_subpile_coverage > find_histo_height(j, overlap_begins, overlap_ends) * q) {
                    first_valid_point = j;
                    break;
                }
            }

            slope_regions[i].second = last_valid_point;
            slope_regions[i + 1].first = first_valid_point << 1 | 0;
        }
    }

    return slope_regions;
}

void Pile::find_median(std::vector<uint32_t> &overlap_begins, std::vector<uint32_t> &overlap_ends) {

    int mean = 0;

    for(int i = begin_; i < end_; ++i) {
        mean = mean + find_histo_height(i, overlap_begins, overlap_ends);
    }

    mean = mean / (end_ - begin_);

    // std::vector<uint16_t> valid_data(data_.begin() + begin_, data_.begin() + end_);

    // std::nth_element(valid_data.begin(), valid_data.begin() + valid_data.size() / 2,
    //     valid_data.end());
    // median_ = valid_data[valid_data.size() / 2];

    // std::nth_element(valid_data.begin(), valid_data.begin() + valid_data.size() / 10,
    //     valid_data.end());
    // p10_ = valid_data[valid_data.size() / 10];
}

void Pile::add_layers(std::vector<uint32_t>& overlap_bounds, std::vector<uint32_t> &overlap_begins, std::vector<uint32_t> &overlap_ends) {

    // printf("This begin = %d\n", begin_);
    if (overlap_bounds.empty()) {
        return;
    }

    std::sort(overlap_bounds.begin(), overlap_bounds.end());

    uint16_t coverage = 0;
    uint32_t last_bound = begin_;
    for (const auto& bound: overlap_bounds) {
        if (coverage > 0 ) {
            for (uint32_t i = last_bound; i < (bound >> 1); ++i) {
                data_[i] += coverage;
            }
        }
        last_bound = (bound >> 1);
        if (bound & 1) {
            --coverage;
        } else {
            ++coverage;
        }
    }
}

// Getting rid of shrink
bool Pile::shrink(uint32_t begin, uint32_t end) {

    if (begin > end) {
        fprintf(stderr, "[rala::Pile::shrink] error: "
            "invalid begin, end coordinates!\n");
        exit(1);
    }

    if (end - begin < 1260) {
        return false;
    }

    for (uint32_t i = begin_; i < begin; ++i) {
        data_[i] = 0;
    }
    begin_ = begin;

    for (uint32_t i = end; i < end_; ++i) {
        data_[i] = 0;
    }
    end_ = end;

    return true;
}

/*
Starting CUDA implementation
Notes:  Depends on 
            -- data_ (or overlap begins/ends)
            -- begin_ and end_
        Must return new_begin and new_end and set this particular piles' begin_ and end_to that
*/
bool Pile::find_valid_region(std::vector<uint32_t> &overlap_begins, std::vector<uint32_t> &overlap_ends) {

    //printf("find_valid_region from %d to %d\n", seq_begin_, seq_end_);
    uint32_t new_begin = 0, new_end = 0, current_begin = 0;
    bool found_begin = false;
    for (uint32_t i = begin_; i < end_; ++i) {
        // Using height function here. Saving as variable because we use it twice
        uint32_t this_height = find_histo_height(i, overlap_begins, overlap_ends);
        if (!found_begin && this_height >= 4) {
            current_begin = i;
            found_begin = true;
        } else if (found_begin && this_height < 4) {
            if (i - current_begin > new_end - new_begin) {
                new_begin = current_begin;
                new_end = i;
            }
            found_begin = false;
        }
    }
    if (found_begin) {
        if (end_ - current_begin > new_end - new_begin) {
            new_begin = current_begin;
            new_end = end_;
        }
    }

    // Will eventually copy shrink code here once we're rid of data_ element
    return shrink(new_begin, new_end);
}

void Pile::find_chimeric_pits(std::vector<uint32_t> &overlap_begins, std::vector<uint32_t> &overlap_ends) {

    auto slope_regions = find_slopes(1.82, overlap_begins, overlap_ends);
    if (slope_regions.empty()) {
        return;
    }

    for (uint32_t i = 0; i < slope_regions.size() - 1; ++i) {
        if (!(slope_regions[i].first & 1) && (slope_regions[i + 1].first & 1)) {
            chimeric_pits_.emplace_back(slope_regions[i].first >> 1,
                slope_regions[i + 1].second);
        }
    }
    intervalMerge(chimeric_pits_);
}

bool Pile::break_over_chimeric_pits(uint16_t dataset_median) {

    auto is_chimeric_pit = [&](uint32_t begin, uint32_t end) -> bool {
        for (uint32_t i = begin; i <= end; ++i) {
            if (data_[i] * 1.84 <= dataset_median) {
                return true;
            }
        }
        return false;
    };

    uint32_t begin = 0, end = 0, last_begin = this->begin_;
    std::vector<std::pair<uint32_t, uint32_t>> tmp;

    for (const auto& it: chimeric_pits_) {
        if (begin_ > it.first || end_ < it.second) {
            continue;
        }
        if (is_chimeric_pit(it.first, it.second)) {
            if (it.first - last_begin > end - begin) {
                begin = last_begin;
                end = it.first;
            }
            last_begin = it.second;
        } else {
            tmp.emplace_back(it);
        }
    }
    if (this->end_ - last_begin > end - begin) {
        begin = last_begin;
        end = this->end_;
    }

    chimeric_pits_.swap(tmp);

    return shrink(begin, end);
}

void Pile::find_chimeric_hills(std::vector<uint32_t> &overlap_begins, std::vector<uint32_t> &overlap_ends) {

    auto slope_regions = brute_find_slopes(1.3, overlap_begins, overlap_ends);
    if (slope_regions.empty()) {
        return;
    }

    auto is_chimeric_hill = [&](
        const std::pair<uint32_t, uint32_t>& begin,
        const std::pair<uint32_t, uint32_t>& end) -> bool {

        // If it's at the beginning or the end, return false
        if ((begin.first >> 1) < 0.05 * (this->end_ - this->begin_) + this->begin_ ||
            end.second > 0.95 * (this->end_ - this->begin_) + this->begin_ ||
            (end.first >> 1) - begin.second > 840) {
            return false;
        }

        // 
        uint32_t peak_value = 1.3 * std::max(find_histo_height(begin.second, overlap_begins, overlap_ends),
                                             find_histo_height(end.first >> 1, overlap_begins, overlap_ends));

        for (uint32_t i = begin.second + 1; i < (end.first >> 1); ++i) {
            if (find_histo_height(i, overlap_begins, overlap_ends) > peak_value) {
                return true;
            }
        }
        return false;
    };

    uint32_t fuzz = 420;
    for (uint32_t i = 0; i < slope_regions.size() - 1; ++i) {
        if (!(slope_regions[i].first & 1)) {
            continue;
        }

        for (uint32_t j = i + 1; j < slope_regions.size(); ++j) {
            if (slope_regions[j].first & 1) {
                continue;
            }

            if (is_chimeric_hill(slope_regions[i], slope_regions[j])) {
                uint32_t begin = (slope_regions[i].first >> 1) - this->begin_ > fuzz ?
                    (slope_regions[i].first >> 1) - fuzz : this->begin_;
                uint32_t end = this->end_ - slope_regions[j].second > fuzz ?
                    slope_regions[j].second + fuzz : this->end_;
                chimeric_hills_.emplace_back(begin, end);
            }
        }
    }
    intervalMerge(chimeric_hills_);

    chimeric_hill_coverage_.resize(chimeric_hills_.size(), 0);
}

void Pile::check_chimeric_hills(const std::unique_ptr<Overlap>& overlap) {

    uint32_t begin = this->begin_ + (overlap->a_id() == id_ ? overlap->a_begin() :
        overlap->b_begin());
    uint32_t end = this->begin_ + (overlap->a_id() == id_ ? overlap->a_end() :
        overlap->b_end());

    for (uint32_t i = 0; i < chimeric_hills_.size(); ++i) {
        if (begin < chimeric_hills_[i].first && end > chimeric_hills_[i].second) {
            ++chimeric_hill_coverage_[i];
        }
    }
}

bool Pile::break_over_chimeric_hills() {

    uint32_t begin = 0, end = 0, last_begin = this->begin_;

    for (uint32_t i = 0; i < chimeric_hills_.size(); ++i) {
        if (begin_ > chimeric_hills_[i].first || end_ < chimeric_hills_[i].second) {
            continue;
        }
        if (chimeric_hill_coverage_[i] > 3) {
            continue;
        }

        if (chimeric_hills_[i].first - last_begin > end - begin) {
            begin = last_begin;
            end = chimeric_hills_[i].first;
        }
        last_begin = chimeric_hills_[i].second;
    }
    if (this->end_ - last_begin > end - begin) {
        begin = last_begin;
        end = this->end_;
    }

    std::vector<std::pair<uint32_t, uint32_t>>().swap(chimeric_hills_);
    std::vector<uint32_t>().swap(chimeric_hill_coverage_);

    return shrink(begin, end);
}

void Pile::find_repetitive_hills(uint16_t dataset_median, std::vector<uint32_t> &overlap_begins, std::vector<uint32_t> &overlap_ends) {

    // // TODO: remove?
    // if (median_ > 1.42 * dataset_median) {
    //     dataset_median = std::max(dataset_median, p10_);
    // }

    auto slope_regions = find_slopes(1.42, overlap_begins, overlap_ends);
    if (slope_regions.empty()) {
        return;
    }

    auto is_repeat_hill = [&](
        const std::pair<uint32_t, uint32_t>& begin,
        const std::pair<uint32_t, uint32_t>& end) -> bool {

        if (((end.first >> 1) + end.second) / 2 -
            ((begin.first >> 1) + begin.second) / 2 > 0.84 * (this->end_ - this->begin_)) {
            return false;
        }
        bool found_peak = false;
        uint32_t peak_value = 1.42 * std::max(find_histo_height(begin.second, overlap_begins, overlap_ends), 
                                              find_histo_height(end.first >> 1, overlap_begins, overlap_ends));
        uint32_t valid_points = 0;
        uint32_t min_value = dataset_median * 1.42;

        for (uint32_t i = begin.second + 1; i < (end.first >> 1); ++i) {
            uint32_t this_height = find_histo_height(i, overlap_begins, overlap_ends);
            if (this_height > min_value) {
                ++valid_points;
            }
            if (this_height > peak_value) {
                found_peak = true;
            }
        }

        if (!found_peak || valid_points < 0.9 * ((end.first >> 1) - begin.second)) {
            return false;
        }
        return true;
    };

    for (uint32_t i = 0; i < slope_regions.size() - 1; ++i) {
        if (!(slope_regions[i].first & 1)) {
            continue;
        }
        for (uint32_t j = i + 1; j < slope_regions.size(); ++j) {
            if (slope_regions[j].first & 1) {
                continue;
            }

            if (is_repeat_hill(slope_regions[i], slope_regions[j])) {
                repeat_hills_.emplace_back(
                    slope_regions[i].second - 0.336 *
                        (slope_regions[i].second - (slope_regions[i].first >> 1)),
                    (slope_regions[j].first >> 1) + 0.336 *
                        (slope_regions[j].second - (slope_regions[j].first >> 1)));
            }
        }
    }

    intervalMerge(repeat_hills_);
    for (auto& it: repeat_hills_) {
        it.first = std::max(begin_, it.first);
        it.second = std::min(end_, it.second);
    }

    repeat_hill_coverage_.resize(repeat_hills_.size(), false);
}

void Pile::check_repetitive_hills(const std::unique_ptr<Overlap>& overlap) {

    uint32_t begin = overlap->b_begin();
    uint32_t end = overlap->b_end();
    uint32_t fuzz = 420;

    for (uint32_t i = 0; i < repeat_hills_.size(); ++i) {
        if (begin < repeat_hills_[i].second && repeat_hills_[i].first < end) {
            if (repeat_hills_[i].first < 0.1 * (this->end_ - this->begin_) + this->begin_ &&
                begin - this->begin_ < this->end_ - end) {
                // left hill
                if (end >= repeat_hills_[i].second + fuzz) {
                    repeat_hill_coverage_[i] = true;
                }
            } else if (repeat_hills_[i].second > 0.9 * (this->end_ - this->begin_) + this->begin_ &&
                begin - this->begin_ > this->end_ - end) {
                // right hill
                if (begin + fuzz <= repeat_hills_[i].first) {
                    repeat_hill_coverage_[i] = true;
                }
            }

        }
    }
}

void Pile::add_repetitive_region(uint32_t begin, uint32_t end) {

    if (begin > end_ || end > end_) {
        fprintf(stderr, "[rala::Pile::add_repetitive_region] error: "
            "[begin,end] out of bounds!\n");
        exit(1);
    }

    repeat_hills_.emplace_back(begin, end);
}

bool Pile::is_valid_overlap(uint32_t begin, uint32_t end) const {

    uint32_t fuzz = 420;

    auto check_hills = [&](const std::vector<std::pair<uint32_t, uint32_t>>& hills) -> bool {
        for (uint32_t i = 0; i < hills.size(); ++i) {
            const auto& it = hills[i];
            if (begin < it.second && it.first < end) {
                if (it.first < 0.1 * (this->end_ - this->begin_) + this->begin_) {
                    // left hill
                    if (end < it.second + fuzz && repeat_hill_coverage_[i]) {
                        return false;
                    }
                } else if (it.second > 0.9 * (this->end_ - this->begin_) + this->begin_) {
                    // right hill
                    if (begin + fuzz > it.first && repeat_hill_coverage_[i]) {
                        return false;
                    }
                }
            }
        }
        return true;
    };

    return check_hills(repeat_hills_);
}

std::string Pile::to_json() const {

    std::stringstream ss;
    ss << "\"" << id_ << "\":{";

    ss << "\"y\":[";
    for (uint32_t i = 0; i < data_.size(); ++i) {
        ss << data_[i];
        if (i < data_.size() - 1) {
            ss << ",";
        }
    }
    ss << "],";

    ss << "\"b\":" << begin_ << ",";
    ss << "\"e\":" << end_ << ",";

    ss << "\"h\":[";
    for (uint32_t i = 0; i < repeat_hills_.size(); ++i) {
        ss << repeat_hills_[i].first << "," << repeat_hills_[i].second;
        if (i < repeat_hills_.size() - 1) {
            ss << ",";
        }
    }
    ss << "],";

    ss << "\"m\":" << median_ << ",";
    ss << "\"p10\":" << p10_;
    ss << "}";

    return ss.str();
}
}
