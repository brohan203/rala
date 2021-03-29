/*!
 * @file pile.hpp
 *
 * @brief Pile class header file
 */

#pragma once

#include <stdint.h>
#include <memory>
#include <string>
#include <vector>

namespace rala {

class Overlap;

class Pile;
std::unique_ptr<Pile> createPile(uint64_t id, uint32_t sequence_length);
std::unique_ptr<Pile> createRoPile(uint32_t begin, uint64_t id, uint32_t sequence_length);


class Pile {
public:
    ~Pile() {};

    uint64_t id() const {
        return id_;
    }

    /*!
     * @brief Returns begin_ of the valid interval [begin_, end_>
     */
    uint32_t seq_begin() const {
        return seq_begin_;
    }

    /*!
     * @brief Returns begin_ of the valid interval [begin_, end_>
     */
    uint32_t begin() const {
        return begin_;
    }

    /*!
     * @brief Returns end_ of the valid interval [begin_, end_>
     */
    uint32_t end() const {
        return end_;
    }

    uint16_t p10() const {
        return p10_;
    }

    uint16_t median() const {
        return median_;
    };

    // Binary search to find overlaps that start before or overlaps that end before position
    uint32_t binary_search(std::vector<uint32_t> &overlaps, uint32_t left, uint32_t right, const uint32_t location);

    /*!
     * @brief Finds height of histogram given overlap begins and ends as vectors of uint32_t
     */
    uint32_t find_histo_height(uint32_t location, std::vector<uint32_t> &overlap_begins, std::vector<uint32_t> &overlap_ends);

    /*!
     * @brief Finds height of histogram given overlap begins and ends as vectors of pair<uint32_t, uint32_t>
     */
    uint32_t new_find_histo_height(uint32_t location, const std::vector<std::pair<uint32_t, uint32_t>> &new_overlap_begins, const std::vector<std::pair<uint32_t, uint32_t>> &new_overlap_ends);

    void find_median();

    const std::vector<uint16_t>& data() const {
        return data_;
    }

    /*!
     * @brief Fills data_ with zeroes in interval [begin_, end_]
     */
    void clear() {
        std::fill(data_.begin() + begin_, data_.begin() + end_, 0);
    }

    /*!
     * @brief Adds overlaps to data_
     */
    void add_layers(std::vector<uint32_t>& overlap_bounds, std::vector<uint32_t> &overlap_begins, std::vector<uint32_t> &overlap_ends);

    /*!
     * @brief Sets values of data_ outside the interval [begin, end> to zeroes
     * and updates begin_, end_ accordingly
     */
    bool shrink(uint32_t begin, uint32_t end);

    /*!
     * @brief Locates region in data_ with values greater or equal to predefined
     * coverage; updates begin_, end_ and data_ accordingly;
     * if there is no such region (with valid coverage and longer than 1260),
     * false is returned
     */
    bool find_valid_region(std::vector<uint32_t> &new_overlap_begins, std::vector<uint32_t> &new_overlap_ends);

    /*!
     * @brief Locates chimeric pits (coverage drops) in data_
     */
    void find_chimeric_pits();

    bool has_chimeric_pit() const {
        return !chimeric_pits_.empty();
    }

    /*!
     * @brief Truncates data_ to longest region without chimeric pits
     */
    bool break_over_chimeric_pits(uint16_t dataset_median);

    /*!
     * @brief Locates possible chimeric hills in data_
     */
    void find_chimeric_hills();

    bool has_chimeric_hill() const {
        return !chimeric_hills_.empty();
    }

    /*!
     * @brief Adds coverage to chimeric hills to decrease false positives
     */
    void check_chimeric_hills(const std::unique_ptr<Overlap>& overlap);

    /*!
     * @brief Truncates data_ to longest region without chimeric hills
     */
    bool break_over_chimeric_hills();

    bool has_chimeric_region() const {
        return has_chimeric_hill() || has_chimeric_pit();
    }

    /*!
     * @brief Locates regions in data_ which ought to be repetitive in the
     * genome and stores them in repeat_hills_
     */
    void find_repetitive_hills(uint16_t dataset_median);

    bool has_repetitive_hills() const {
        return !repeat_hills_.empty();
    }

    /*!
     * @brief Adds coverage to repetitive hill (number of reads passing through)
     */
    void check_repetitive_hills(const std::unique_ptr<Overlap>& overlap);

    /*
     * @brief Manually add repetitive region in data_
     */
    void add_repetitive_region(uint32_t begin, uint32_t end);

    /*!
     * @brief Checks whether overlap [begin, end> is valid with respect to
     * hills_ which indicate repetitive regions of the genome
     */
    bool is_valid_overlap(uint32_t begin, uint32_t end) const;

    /*!
     * @brief Serializes objects into JSON format
     */
    std::string to_json() const;

    friend std::unique_ptr<Pile> createPile(uint64_t id, uint32_t sequence_length);
    friend std::unique_ptr<Pile> createRoPile(uint32_t begin, uint64_t id, uint32_t sequence_length);
private:
    Pile(uint64_t id, uint32_t sequence_length);
    Pile(uint32_t begin, uint64_t id, uint32_t sequence_length);
    Pile(const Pile&) = delete;
    const Pile& operator=(const Pile&) = delete;

    std::vector<std::pair<uint32_t, uint32_t>> find_slopes(double q);

    uint64_t id_;
    uint32_t begin_;
    uint32_t end_;
    uint32_t seq_begin_;
    uint32_t seq_end_;
    uint16_t p10_;
    uint16_t median_;
    std::vector<uint16_t> data_;
    std::vector<std::pair<uint32_t, uint32_t>> repeat_hills_;
    std::vector<bool> repeat_hill_coverage_;
    std::vector<std::pair<uint32_t, uint32_t>> chimeric_pits_;
    std::vector<std::pair<uint32_t, uint32_t>> chimeric_hills_;
    std::vector<uint32_t> chimeric_hill_coverage_;
};

}
