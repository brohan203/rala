/*!
 * @file graph.cpp
 *
 * @brief Graph class source file
 */

#include <set>
#include <list>
#include <stack>
#include <deque>
#include <math.h>

#include "read.hpp"
#include "overlap.hpp"
#include "graph.hpp"

namespace RALAY {

constexpr uint32_t kMinOverlapLength = 2000;
constexpr uint32_t kMinMatchingBases = 100;
constexpr double kMinMatchingBasesPerc = 0.05;
constexpr uint32_t kMinCoverage = 3;
constexpr uint32_t kMaxOverhang = 1000;
constexpr double kMaxOverhangToOverlapRatio = 0.8;
constexpr double kTransitiveEdgeEps = 0.12;
constexpr double kShortLongOverlapRatio = 0.2;
constexpr uint32_t kMaxBubbleLength = 250000;
constexpr double kBubbleCoverageEps = 0.05;
constexpr uint32_t kMinUnitigSize = 5;

static bool isSimilar(double a, double b, double eps) {
    return (a >= b * (1 - eps) && a <= b * (1 + eps));
};

void calculateReadCoverages(std::vector<std::shared_ptr<Read>>& reads,
    const std::vector<std::shared_ptr<Overlap>>& overlaps) {

    std::vector<uint32_t> coverage(reads.size(), 0);
    for (const auto& overlap: overlaps) {

        if (overlap->a_end() - overlap->a_begin() < kMinOverlapLength ||
            overlap->b_end() - overlap->b_begin() < kMinOverlapLength ||
            overlap->matching_bases() < kMinMatchingBases ||
            overlap->quality() < kMinMatchingBasesPerc) {
            continue;
        }

        uint32_t begin = (overlap->a_rc() ? overlap->a_length() - overlap->a_end() : overlap->a_begin());
        uint32_t end = (overlap->a_rc() ? overlap->a_length() - overlap->a_begin() : overlap->a_end());
        coverage[overlap->a_id()] += end - begin;

        begin = overlap->b_rc() ? overlap->b_length() - overlap->b_end() : overlap->b_begin();
        end = overlap->b_rc() ? overlap->b_length() - overlap->b_begin() : overlap->b_end();
        coverage[overlap->b_id()] += end - begin;
    }

    for (uint32_t i = 0; i < coverage.size(); ++i) {
        reads[i]->set_coverage(coverage[i] / (double) reads[i]->sequence().size());
    }
}

void trimReads(std::vector<std::shared_ptr<Read>>& reads,
    std::vector<std::shared_ptr<Overlap>>& overlaps) {

    std::vector<std::vector<uint32_t>> coverage(reads.size());
    uint32_t tot = 0, rtot = 0;
    for (const auto& overlap: overlaps) {

        if (overlap->a_end() - overlap->a_begin() < kMinOverlapLength ||
            overlap->b_end() - overlap->b_begin() < kMinOverlapLength ||
            overlap->matching_bases() < kMinMatchingBases) {
            continue;
        }
        ++tot;
        if (overlap->quality() < kMinMatchingBasesPerc) {
            continue;
        }

        if (coverage[overlap->a_id()].size() == 0) {
            coverage[overlap->a_id()].resize(overlap->a_length(), 0);
        }
        uint32_t begin = (overlap->a_rc() ? overlap->a_length() - overlap->a_end() : overlap->a_begin());
        uint32_t end = (overlap->a_rc() ? overlap->a_length() - overlap->a_begin() : overlap->a_end());

        for (uint32_t i = begin; i < end; ++i) {
            ++coverage[overlap->a_id()][i];
        }

        if (coverage[overlap->b_id()].size() == 0) {
            coverage[overlap->b_id()].resize(overlap->b_length(), 0);
        }
        begin = overlap->b_rc() ? overlap->b_length() - overlap->b_end() : overlap->b_begin();
        end = overlap->b_rc() ? overlap->b_length() - overlap->b_begin() : overlap->b_end();

        for (uint32_t i = begin; i < end; ++i) {
            ++coverage[overlap->b_id()][i];
        }
    }
    fprintf(stderr, "\n");
    fprintf(stderr, "Totaly: %u\n", tot);

    fprintf(stderr, "Regions done\n");

    for (uint32_t c = 0; c < coverage.size(); ++c) {
        const auto& cov = coverage[c];
        if (!cov.empty()) {
            // find longest region with coverage at least kMinCoverage
            std::vector<uint32_t> regions;
            for (uint32_t i = 0; i < cov.size(); ++i) {
                if (cov[i] >= kMinCoverage) {
                    regions.push_back(i);
                    for (uint32_t j = i + 1; j < cov.size(); ++j, ++i) {
                        if (cov[j] < kMinCoverage) {
                            regions.push_back(j);
                            break;
                        }
                    }
                }
            }
            if (regions.size() % 2 != 0) {
                regions.push_back(cov.size());
            }

            if (!regions.empty()) {
                fprintf(stderr, "%d:", c + 1);
                for (const auto& it: regions) {
                    fprintf(stderr, " %u", it);
                }
                fprintf(stderr, "\n");
                ++rtot;
                //fprintf(stderr, "%s: %zu ->", reads[c]->name().c_str(), reads[c]->sequence().size());
                reads[c]->trim_sequence(regions.front(), regions.back());
                //fprintf(stderr, "%zu\n", reads[c]->sequence().size());
            }
        }

        printf("@%s\n%s\n+\n%s\n",
            reads[c]->name().c_str(),
            reads[c]->sequence().c_str(),
            reads[c]->quality().c_str());
    }

    fprintf(stderr, "Totaly r: %u\n", rtot);
}

uint32_t classifyOverlap(const std::shared_ptr<Overlap>& overlap) {

    uint32_t overhang = std::min(overlap->a_begin(), overlap->b_begin()) +
        std::min(overlap->a_length() - overlap->a_end(), overlap->b_length() - overlap->b_end());

    if (overhang > std::min(kMaxOverhang, (uint32_t) (kMaxOverhangToOverlapRatio * overlap->length()))) {
        return 0; // internal match
    }
    if (overlap->a_begin() <= overlap->b_begin() && (overlap->a_length() - overlap->a_end()) <= (overlap->b_length() - overlap->b_end())) {
        return 1; // a contained
    }
    if (overlap->a_begin() >= overlap->b_begin() && (overlap->a_length() - overlap->a_end()) >= (overlap->b_length() - overlap->b_end())) {
        return 2; // b contained
    }
    if (overlap->a_begin() > overlap->b_begin()) {
        return 3; // a to b overlap
    }

    return 4; // b to a overlap
}

class Graph::Node {
public:
    // Node encapsulating read
    Node(uint32_t _id, const std::shared_ptr<Read>& read) :
            id(_id), read_id(read->id()), pair(), sequence(id % 2 == 0 ? read->sequence() : read->rc()),
            prefix_edges(), suffix_edges(), scores(1, read->coverage()), mark(false) {
    }
    // Unitig
    Node(uint32_t _id, Node* begin_node, Node* end_node);
    Node(const Node&) = delete;
    const Node& operator=(const Node&) = delete;

    ~Node() {}

    uint32_t length() const {
        return sequence.size();
    }

    uint32_t in_degree() const {
        return prefix_edges.size();
    }

    uint32_t out_degree() const {
        return suffix_edges.size();
    }

    uint32_t unitig_size() const {
        return scores.size();
    }

    double coverage() const {
        double cov = 0.0;
        for (const auto& it: scores) cov += it;
        return cov / scores.size();
    }

    bool is_junction() const {
        return (out_degree() > 1 || in_degree() > 1);
    }

    bool is_tip() const {
        return (out_degree() == 0 || in_degree() == 0) && unitig_size() < kMinUnitigSize;
    }

    uint32_t id;
    uint32_t read_id;
    Node* pair;
    std::string sequence;
    std::list<Edge*> prefix_edges;
    std::list<Edge*> suffix_edges;
    std::vector<double> scores;
    bool mark;
};

class Graph::Edge {
public:
    Edge(uint32_t _id, const std::shared_ptr<Overlap>& overlap, Node* _begin_node,
        Node* _end_node, uint32_t type) :
            id(_id), pair(), begin_node(_begin_node), end_node(_end_node), length(),
            score(overlap->quality()), mark(false) {

        uint32_t length_a = id % 2 == 0 ? overlap->a_begin() : overlap->a_length() - overlap->a_end();
        uint32_t length_b = id % 2 == 0 ? overlap->b_begin() : overlap->b_length() - overlap->b_end();

        if (type == 0) { // a to b overlap
            length = length_a - length_b;
        } else { // b to a overlap
            length = length_b - length_a;
        }
    }
    Edge(const Edge&) = delete;
    const Edge& operator=(const Edge&) = delete;

    ~Edge() {}

    std::string label() const {
        return begin_node->sequence.substr(0, length);
    }

    uint32_t quality() const {
        return (score * (begin_node->length() - length));
    }

    uint32_t id;
    Edge* pair;
    Node* begin_node;
    Node* end_node;
    uint32_t length;
    double score;
    bool mark;
};

Graph::Node::Node(uint32_t _id, Node* begin_node, Node* end_node) :
        id(_id), read_id(), pair(), sequence(), prefix_edges(), suffix_edges(), scores(), mark(false) {

    if (!begin_node->prefix_edges.empty()) {
        begin_node->prefix_edges.front()->end_node = this;
        prefix_edges.push_back(begin_node->prefix_edges.front());
    }

    uint32_t length = 0;
    Node* curr_node = begin_node;
    while (curr_node->id != end_node->id) {
        auto* edge = curr_node->suffix_edges.front();
        edge->mark = true;

        scores.insert(scores.begin(), curr_node->scores.begin(), curr_node->scores.end());
        length += edge->length;
        sequence += edge->label();

        curr_node->prefix_edges.clear();
        curr_node->suffix_edges.clear();
        curr_node->mark = true;

        curr_node = edge->end_node;
    }

    scores.insert(scores.begin(), end_node->scores.begin(), end_node->scores.end());
    sequence += end_node->sequence;

    if (!end_node->suffix_edges.empty()) {
        end_node->suffix_edges.front()->begin_node = this;
        end_node->suffix_edges.front()->length += length;
        suffix_edges.push_back(end_node->suffix_edges.front());
    }

    end_node->prefix_edges.clear();
    end_node->suffix_edges.clear();
    end_node->mark = true;
}

std::unique_ptr<Graph> createGraph(const std::vector<std::shared_ptr<Read>>& reads,
    const std::vector<std::shared_ptr<Overlap>>& overlaps) {
    return std::unique_ptr<Graph>(new Graph(reads, overlaps));
}

Graph::Graph(const std::vector<std::shared_ptr<Read>>& reads,
    const std::vector<std::shared_ptr<Overlap>>& overlaps) :
        nodes_(), edges_() {

    // remove contained reads and their overlaps before graph construction
    uint32_t max_read_id = 0;
    for (const auto& read: reads) {
        max_read_id = std::max(max_read_id, read->id());
    }

    std::vector<bool> is_contained(max_read_id + 1, false);
    std::vector<uint8_t> overlap_type(overlaps.size(), 0);

    for (uint32_t i = 0; i < overlaps.size(); ++i) {
        overlap_type[i] = classifyOverlap(overlaps[i]);

        if (overlap_type[i] == 1) { // a contained
            is_contained[overlaps[i]->a_id()] = true;
        } else if (overlap_type[i] == 2) { // b contained
            is_contained[overlaps[i]->b_id()] = true;
        }
    }

    // create assembly graph
    std::vector<int32_t> read_id_to_node_id(max_read_id + 1, -1);
    uint32_t node_id = 0;
    for (const auto& read: reads) {
        if (!is_contained[read->id()]) {
            read_id_to_node_id[read->id()] = node_id;

            Node* node = new Node(node_id++, read); // normal read
            Node* _node = new Node(node_id++, read); // reverse complement

            node->pair = _node;
            _node->pair = node;

            nodes_.push_back(std::unique_ptr<Node>(node));
            nodes_.push_back(std::unique_ptr<Node>(_node));
        }
    }

    uint32_t edge_id = 0;
    for (uint32_t i = 0; i < overlaps.size(); ++i) {
        const auto& overlap = overlaps[i];
        if (!is_contained[overlap->a_id()] && !is_contained[overlap->b_id()]) {
            if (overlap_type[i] < 3) {
                continue;
            }

            if (overlap->a_end() - overlap->a_begin() < kMinOverlapLength ||
                overlap->b_end() - overlap->b_begin() < kMinOverlapLength ||
                overlap->matching_bases() < kMinMatchingBases) {
                continue;
            }

            auto a = nodes_[read_id_to_node_id[overlap->a_id()] + (overlap->a_rc() == 0 ? 0 : 1)].get();
            auto _a = a->pair;

            auto b = nodes_[read_id_to_node_id[overlap->b_id()] + (overlap->b_rc() == 0 ? 0 : 1)].get();
            auto _b = b->pair;

            if (overlap_type[i] == 3) { // a to b overlap
                Edge* edge = new Edge(edge_id++, overlap, a, b, 0);
                Edge* _edge = new Edge(edge_id++, overlap, _b, _a, 1);

                edge->pair = _edge;
                _edge->pair = edge;

                edges_.push_back(std::unique_ptr<Edge>(edge));
                edges_.push_back(std::unique_ptr<Edge>(_edge));

                a->suffix_edges.push_back(edge);
                _a->prefix_edges.push_back(_edge);
                b->prefix_edges.push_back(edge);
                _b->suffix_edges.push_back(_edge);

            } else if (overlap_type[i] == 4) { // b to a overlap
                Edge* edge = new Edge(edge_id++, overlap, b, a, 1);
                Edge* _edge = new Edge(edge_id++, overlap, _a, _b, 0);

                edge->pair = _edge;
                _edge->pair = edge;

                edges_.push_back(std::unique_ptr<Edge>(edge));
                edges_.push_back(std::unique_ptr<Edge>(_edge));

                b->suffix_edges.push_back(edge);
                _b->prefix_edges.push_back(_edge);
                a->prefix_edges.push_back(edge);
                _a->suffix_edges.push_back(_edge);
            }
        }
    }
    fprintf(stderr, "NODES = %zu, HITS = %zu\n", nodes_.size(), edges_.size());
}

Graph::~Graph() {
}

void Graph::remove_isolated_nodes() {

    for (auto& node: nodes_) {
        if (node == nullptr) {
            continue;
        }
        if ((node->in_degree() == 0 && node->out_degree() == 0 && node->unitig_size() < kMinUnitigSize) || (node->mark == true)) {
            // fprintf(stderr, "Removing isolated node: %d\n", node->id);
            node.reset();
        }
    }
}

void Graph::remove_transitive_edges() {

    std::vector<Edge*> candidate_edge(nodes_.size(), nullptr);

    for (const auto& node_x: nodes_) {
        if (node_x == nullptr) continue;

        for (const auto& edge: node_x->suffix_edges) {
            candidate_edge[edge->end_node->id] = edge;
        }

        for (const auto& edge_xy: node_x->suffix_edges) {
            for (const auto& edge_yz: nodes_[edge_xy->end_node->id]->suffix_edges) {
                uint32_t z = edge_yz->end_node->id;
                if (candidate_edge[z] != nullptr && candidate_edge[z]->mark == false) {
                    if (isSimilar(edge_xy->length + edge_yz->length, candidate_edge[z]->length, kTransitiveEdgeEps)) {
                        candidate_edge[z]->mark = true;
                        candidate_edge[z]->pair->mark = true;
                    }
                }
            }
        }

        for (const auto& edge: node_x->suffix_edges) {
            candidate_edge[edge->end_node->id] = nullptr;
        }
    }

    remove_marked_edges();
}

void Graph::remove_long_edges() {

    for (const auto& node: nodes_) {
        if (node == nullptr) continue;

        for (const auto& edge1: node->suffix_edges) {
            for (const auto& edge2: node->suffix_edges) {
                if (edge1->id == edge2->id || edge1->mark == true || edge2->mark == true) continue;
                if (edge1->quality() > edge2->quality()) {
                    if (edge2->quality() / (double) edge1->quality() < kShortLongOverlapRatio) {
                        fprintf(stderr, "Removing :OOOO\n");
                        edge2->mark = true;
                        edge2->pair->mark = true;
                    }
                }
            }
        }
    }

    remove_marked_edges();
}

void Graph::remove_tips() {

    for (const auto& node: nodes_) {
        if (node == nullptr || !node->is_tip()) continue;

        bool is_tip = true;
        // bool is_only_tip = true;
        if (node->in_degree() == 0) {
            /*for (const auto& edge: node->suffix_edges) {
                if (edge->end_node->in_degree() < 2) {
                    is_tip = false;
                    break;
                } else {
                    for (const auto& edge2: edge->end_node->prefix_edges) {
                        if (edge2->begin_node->id == node->id) continue;
                        if (edge2->begin_node->is_tip()) {
                            is_only_tip = false;
                            if (edge2->begin_node->unitig_size() < node->unitig_size()) {
                                is_tip = false;
                                break;
                            }
                        }
                    }
                }
            }*/
            if (is_tip) { //&& (is_only_tip || node->unitig_size() < kMinUnitigSize)) {
                for (auto& edge: node->suffix_edges) {
                    edge->mark = true;
                    edge->pair->mark = true;
                }
                node->mark = true;
                node->pair->mark = true;
            }
        } else if (node->out_degree() == 0) {
            /*for (const auto& edge: node->prefix_edges) {
                if (edge->begin_node->out_degree() < 2) {
                    is_tip = false;
                    break;
                } else {
                   for (const auto& edge2: edge->begin_node->suffix_edges) {
                       if (edge2->end_node->id == node->id) continue;
                       if (edge2->end_node->is_tip()) {
                           is_only_tip = false;
                           if (edge2->end_node->unitig_size() < node->unitig_size()) {
                               is_tip = false;
                               break;
                           }
                       }
                   }
               }
           }*/
            if (is_tip) { //&& (is_only_tip || node->unitig_size() < kMinUnitigSize)) {
                for (auto& edge: node->prefix_edges) {
                    edge->mark = true;
                    edge->pair->mark = true;
                }
                node->mark = true;
                node->pair->mark = true;
            }
        }

        remove_marked_edges();
    }

    remove_isolated_nodes();
}

void Graph::remove_cycles() {

    std::stack<uint32_t> stack;
    std::vector<int32_t> indexes(nodes_.size(), -1);
    std::vector<int32_t> low_links(nodes_.size(), -1);
    std::vector<bool> is_on_stack(nodes_.size(), false);
    int32_t index = 0;

    std::vector<std::vector<uint32_t>> cycles;

    std::function<void(uint32_t)> strong_connect = [&](uint32_t v) -> void {
        indexes[v] = index;
        low_links[v] = index;
        ++index;
        // fprintf(stderr, "Pushing %d\n", v);
        stack.push(v);
        is_on_stack[v] = true;

        for (const auto& edge: nodes_[v]->suffix_edges) {
            uint32_t w = edge->end_node->id;
            if (indexes[w] == -1) {
                strong_connect(w);
                low_links[v] = std::min(low_links[v], low_links[w]);
            } else if (is_on_stack[w]) {
                low_links[v] = std::min(low_links[v], indexes[w]);
            }
        }

        if (low_links[v] == indexes[v]) {
            // new strongly connected component
            std::vector<uint32_t> scc = { v };
            uint32_t w;
            do {
                w = stack.top();
                stack.pop();
                is_on_stack[w] = false;
                scc.push_back(w);
            } while (v != w);

            if (scc.size() > 2) {
                cycles.push_back(scc);
            }
        }
    };

    do {
        cycles.clear();
        for (const auto& node: nodes_) {
            if (node == nullptr) continue;
            if (indexes[node->id] == -1) {
                strong_connect(node->id);
            }
        }

        fprintf(stderr, "Number of cycles %zu\n", cycles.size());

        for (const auto& cycle: cycles) {

            Edge* worst_edge = nullptr;
            double min_score = 5;

            for (uint32_t i = 0; i < cycle.size() - 1; ++i) {
                const auto& node = nodes_[cycle[i]];
                for (auto& edge: node->prefix_edges) {
                    if (edge->begin_node->id == cycle[i + 1]) {
                        if (min_score > edge->score) {
                            min_score = edge->score;
                            worst_edge = edge;
                        }
                        break;
                    }
                }
            }

            worst_edge->mark = true;
            worst_edge->pair->mark = true;
        }

        remove_marked_edges();

    } while (cycles.size() != 0);
}

void Graph::remove_bubbles() {

    uint32_t bubble_id = 0;

    for (const auto& node: nodes_) {
        if (node == nullptr || node->out_degree() < 2) continue;

        // if (bubble_id == 1) break;

        std::vector<uint32_t> distance(nodes_.size(), -1);
        distance[node->id] = 0;
        std::vector<bool> visited(nodes_.size(), false);

        std::deque<uint32_t> queue;
        queue.push_back(node->id);

        int32_t end_id = -1;

        // BFS
        while (queue.size() != 0) {
            const auto& curr_node = nodes_[queue.front()];
            queue.pop_front();

            uint32_t v = curr_node->id;
            if (visited[v] == true) {
                // found end
                end_id = v;
                break;
            }

            visited[v] = true;
            for (const auto& edge: curr_node->suffix_edges) {
                uint32_t w = edge->end_node->id;

                if (w == node->id) {
                    // Cycle
                    continue;
                }
                if (distance[v] + edge->length > kMaxBubbleLength) {
                    // Out of reach
                    continue;
                }

                distance[w] = distance[v] + edge->length;
                //if (nodes_[w]->out_degree() != 0) {
                    queue.push_back(w);
                //}
            }
        }

        if (end_id == -1) {
            // no bubble found
            continue;
        }

        // backtrack from end node
        queue.clear();
        queue.push_back(end_id);
        std::fill(visited.begin(), visited.end(), false);
        uint32_t begin_id;

        // BFS
        while (queue.size() != 0) {
            const auto& curr_node = nodes_[queue.front()];
            queue.pop_front();

            uint32_t v = curr_node->id;
            if (visited[v] == true) {
                // found begin node;
                begin_id = v;
                break;
            }

            visited[v] = true;
            for (const auto& edge: curr_node->prefix_edges) {
                uint32_t w = edge->begin_node->id;
                if (visited[w] == true) {
                    queue.push_front(w);
                    break;
                }
                queue.push_back(w);
            }
        }

        // fprintf(stderr, "Bubble (search from %d): %d -> %d\n", node->id, begin_id, end_id);
        if (begin_id == (uint32_t) end_id) {
            continue;
        }

        ++bubble_id;

        // find paths between begin & end nodes
        std::vector<std::vector<uint32_t>> paths;
        std::vector<uint32_t> path;
        path.push_back(begin_id);
        std::vector<bool> visited_edge(edges_.size(), false);

        // DFS
        while (path.size() != 0) {

            const auto& curr_node = nodes_[path.back()];
            uint32_t v = curr_node->id;

            if (v == (uint32_t) end_id) {
                paths.push_back(path);
                path.pop_back();
                continue;
            }

            bool valid = false;

            for (const auto& edge: curr_node->suffix_edges) {
                if (visited_edge[edge->id] == false) {
                    path.push_back(edge->end_node->id);
                    visited_edge[edge->id] = true;
                    valid = true;
                    break;
                }
            }

            if (!valid) {
                path.pop_back();
            }
        }

        /*for (const auto& path: paths) {
            for (const auto& v: path) {
                fprintf(stderr, "%u ", v);
            }
            fprintf(stderr, "\n");
        }*/

        // remove the worst path
        double worst_path_coverage = 1000000.0;
        uint32_t worst_path_quality = 0;
        uint32_t worst_path_id = 0;

        for (uint32_t p = 0; p < paths.size(); ++p) {
            double coverage = 0;
            uint32_t quality = 0;
            for (uint32_t i = 0; i < paths[p].size() - 1; ++i) {
                if (i == 0) {
                    for (const auto& edge: node->suffix_edges) {
                        if (edge->end_node->id == paths[p][i+1]) {
                            quality = edge->quality();
                            break;
                        }
                    }
                } else {
                    coverage += nodes_[paths[p][i]]->coverage();
                }
            }

            if (paths[p].size() > 2) {
                coverage /= paths[p].size() - 2;
            }

            if (isSimilar(worst_path_coverage, coverage, kBubbleCoverageEps)) {
                if (worst_path_quality > quality) {
                    worst_path_coverage = coverage;
                    worst_path_quality = quality;
                    worst_path_id = p;
                }
            } else if (worst_path_coverage > coverage) {
                worst_path_coverage = coverage;
                worst_path_quality = quality;
                worst_path_id = p;
            }
        }

        //fprintf(stderr, "Worst path: %d, %g, %d\n", worst_path_id, worst_path_coverage,
        //    worst_path_quality);

        bool has_external_edges = false;
        for (uint32_t i = 0; i < paths[worst_path_id].size() - 1; ++i) {
            const auto& node = nodes_[paths[worst_path_id][i]];
            if (i != 0 && (node->in_degree() > 1 || node->out_degree() > 1)) {
                has_external_edges = true;
                break;
            }
            for (const auto& edge: node->suffix_edges) {
                if (edge->end_node->id == paths[worst_path_id][i+1]) {
                    edge->mark = true;
                    edge->pair->mark = true;
                    break;
                }
            }
            if (has_external_edges) {
                break;
            }
        }

        remove_marked_edges();
    }

    fprintf(stderr, "Bubbles popped %d\n", bubble_id);

    remove_isolated_nodes();
}

void Graph::create_unitigs() {

    uint32_t node_id = nodes_.size();
    std::vector<bool> visited(nodes_.size(), false);
    for (const auto& node: nodes_) {
        if (node == nullptr || visited[node->id] || node->is_junction()) continue;

        auto bnode = node.get();
        while (!bnode->is_junction()) {
            visited[bnode->id] = true;
            visited[bnode->pair->id] = true;
            if (bnode->in_degree() == 0 || bnode->prefix_edges.front()->begin_node->is_junction()) {
                break;
            }
            bnode = bnode->prefix_edges.front()->begin_node;
        }

        auto enode = node.get();
        while (!enode->is_junction()) {
            visited[enode->id] = true;
            visited[enode->pair->id] = true;
            if (enode->out_degree() == 0 || enode->suffix_edges.front()->end_node->is_junction()) {
                break;
            }
            enode = enode->suffix_edges.front()->end_node;
        }

        if (bnode->id == enode->id) {
            continue;
        }

        Node* unitig = new Node(node_id++, bnode, enode); // normal
        Node* _unitig = new Node(node_id++, enode->pair, bnode->pair); // reverse complement

        unitig->pair = _unitig;
        _unitig->pair = unitig;

        nodes_.push_back(std::unique_ptr<Node>(unitig));
        nodes_.push_back(std::unique_ptr<Node>(_unitig));

        fprintf(stderr, "Unitig: %d -> %d && %d -> %d\n", bnode->id, enode->id, enode->pair->id, bnode->pair->id);
    }

    remove_marked_edges();
    remove_isolated_nodes();
}

void Graph::print_contigs() const {

    for (const auto& node: nodes_) {
        if (node == nullptr) continue;
        if (node->id != 178) continue;
        fprintf(stderr, "Start: %d", node->id);

        std::string contig = "";
        auto curr_node = node.get();
        while (curr_node->out_degree() != 0) {
            for (const auto& edge: curr_node->suffix_edges) {
                if (edge->end_node->id != 1026 && edge->end_node->id != 502 && edge->end_node->id != 1302) {
                    contig += edge->label();
                    curr_node = edge->end_node;
                    break;
                }
            }
            fprintf(stderr, " %d", curr_node->id);
        }
        fprintf(stderr, "\n");
        contig += curr_node->sequence;
        fprintf(stderr, "%s\n", contig.c_str());
    }
}

void Graph::remove_marked_edges() {

    auto delete_edges = [&](std::list<Edge*>& edges) -> void {
        auto edge = edges.begin();
        while (edge != edges.end()) {
            if ((*edge)->mark == true) {
                edge = edges.erase(edge);
            } else {
                ++edge;
            }
        }
    };

    for (const auto& node: nodes_) {
        if (node == nullptr) continue;
        delete_edges(node->prefix_edges);
        delete_edges(node->suffix_edges);
    }

    for (auto& edge: edges_) {
        if (edge == nullptr) continue;
        if (edge->mark == true) {
            edge.reset();
        }
    }
}

void Graph::print() const {

    printf("digraph 1 {\n");
    printf("    overlap = scalexy\n");

    for (const auto& node: nodes_) {
        if (node == nullptr) continue;

        printf("    %d [label = \"%u [%u] {%d} U:%d C:%g\"", node->id, node->id, node->length(), node->read_id, node->unitig_size(), node->coverage());
        if (node->id % 2 == 1) {
            printf(", style = filled, fillcolor = brown1]\n");
            printf("    %d -> %d [style = dotted, arrowhead = none]\n", node->id, node->id - 1);
        } else {
            printf("]\n");
        }
    }

    for (const auto& edge: edges_) {
        if (edge == nullptr) continue;
        printf("    %d -> %d [label = \"%d, %g\"]\n", edge->begin_node->id, edge->end_node->id, edge->length, edge->score);
    }

    printf("}\n");
}

}
