use ahash::HashSet;

const INPUT_FILE: &str = "input/08.txt";

pub fn a() -> String {
    let input = std::fs::read_to_string(INPUT_FILE).expect("Input should exist");
    a_with_input(&input, 1000).to_string()
}

fn a_with_input(input: &str, num_connections: usize) -> usize {
    let boxes = parse(input);

    // this is still a little bit N^2 log(n), but N is only 1000 so it's fine (24 ms)
    let mut all_edge_lengths: Vec<(usize, usize, u64)> = (0..boxes.len())
        .flat_map(|i| (i + 1..boxes.len()).map(move |j: usize| (i, j)))
        .map(|(i, j)| (i, j, boxes[i].dist_sq(boxes[j])))
        .collect();

    all_edge_lengths.sort_by_key(|(_i, _j, dist)| *dist);

    let mut known_connections: HashSet<(usize, usize)> = HashSet::default();

    for (i, j, _) in all_edge_lengths.iter().copied().take(num_connections) {
        known_connections.insert((i, j));
    }

    let mut circuits = connected_components(boxes.len(), &known_connections);
    circuits.sort_by_key(|b| std::cmp::Reverse(b.len()));

    assert!(circuits.len() >= 3);

    circuits[0].len() * circuits[1].len() * circuits[2].len()
}

fn connected_components(size: usize, edges: &HashSet<(usize, usize)>) -> Vec<HashSet<usize>> {
    // map from index i to "set of vertices reachable from i"
    let edge_lists: Vec<HashSet<usize>> = (0..size)
        .map(|i: usize| (0..size).filter(|j| edges.contains(&(i, *j)) || edges.contains(&(*j, i))).collect())
        .collect();

    fn dfs(node: usize, traversed: &mut HashSet<usize>, component: &mut HashSet<usize>, edges: &[HashSet<usize>]) {
        if traversed.contains(&node) {
            return;
        }

        traversed.insert(node);
        component.insert(node);

        for j in edges[node].iter().copied() {
            dfs(j, traversed, component, edges);
        }
    }

    let mut traversed = HashSet::default();
    let mut components = Vec::new();

    for i in 0..size {
        if traversed.contains(&i) {
            continue;
        }

        let mut component = HashSet::default();

        dfs(i, &mut traversed, &mut component, &edge_lists);

        components.push(component);
    }

    components
}

pub fn b() -> String {
    let input = std::fs::read_to_string(INPUT_FILE).expect("Input should exist");
    b_with_input(&input).to_string()
}

struct UnionFindTree {
    // map from i to j; if parents[i] == j, it means j is the parent of i; if parents[i] == i,
    // it means i is a root (skipping Option for performance reasons)
    parents: Vec<usize>,

    // map from i to "size of tree at or below i". NOTE: this is ONLY meaningful if `i` is a root
    // node (this is all we need, and due to some other optimizations we made, it's very annoying
    // to maintain anything else)
    component_size: Vec<usize>,

    // tracks the number of connected components of the graph, since it's trivial to do so
    // and helpful for our use case
    num_components: usize,
}

impl UnionFindTree {
    fn new(size: usize) -> Self {
        Self {
            component_size: vec![1; size],
            // everything is a root at the beginning
            parents: (0..size).collect(),
            num_components: size,
        }
    }

    fn get_root(&mut self, node: usize) -> usize {
        let mut root = node;
        let mut parent = self.parents[root];
        while parent != root {
            root = parent;
            parent = self.parents[root];
        }

        // this is called path flattening; in practice it makes get_root run incredibly quickly,
        // in "inverse Ackermann" time (amortized) since, essentially, once you run this once,
        // everything from node to the root get set to point to the root, so anything that
        // is in the same component will, as it walks up the tree, hit on a shared node
        // in the path, and then jump straight to the top.
        let mut cleanup_node = node;
        let mut cleanup_parent = self.parents[cleanup_node];
        while cleanup_parent != root {
            self.parents[cleanup_node] = root;
            cleanup_node = cleanup_parent;
            cleanup_parent = self.parents[cleanup_parent];
        }

        root
    }

    fn is_connected(&self) -> bool {
        self.num_components == 1
    }

    fn add_edge(&mut self, i: usize, j: usize) {
        // Idea: first check if they're already connected to each other; if so, stop
        // Otherwise let parent of i's root be j's root
        let i = self.get_root(i);
        let j = self.get_root(j);

        if i == j {
            return;
        }

        let i_size = self.component_size[i];
        let j_size = self.component_size[j];

        // It doesn't matter for this problem, but just for my own amusement (thanks, Chat)
        // this check ensures that our tree is "balanced enough" to ensure logarithmic performance
        // for get_root (and therefore for all operations)
        //
        // Proof: we maintain a pre/postcondition that, if x < y (that is, y is the parent of x),
        //        then comp(y) >= 2 * comp(x), where comp(y) is the number of (transitive)
        //        descendants of y
        // Proof (of condition): clearly true at start, since nothing has a child.
        //        Then, if we set x < y, we know that comp(y) >= comp(x) before the set;
        //        after the set, comp(x) is unchanged, but comp'(y) = comp(y) + comp(x) >= 2 * comp(x),
        //        as desired.
        // Proof (of performance of get_parent): specifically, we prove that if y is a root
        //        and x << y (that is, x is a descendant of y), then get_parent(x) takes at most
        //        log_2(comp(y)) steps.
        //        To see this, suppose get_parent(x) has k steps to get to y, and observe the chain
        //        x = z0 < z1 < z2 < ... < zk = y. By the condition,
        //        comp(y) = comp(zk) >= 2 * comp(zk-1) >= ... >= 2^k comp(z0) >= 2^k.
        //        Therefore, k <= log_2(comp(y)), as was to be shown.
        if j_size < i_size {
            self.parents[j] = i;
            self.component_size[i] += j_size;
        } else {
            self.parents[i] = j;
            self.component_size[j] += i_size;
        }

        self.num_components -= 1;
    }
}

fn b_with_input(input: &str) -> u64 {
    let boxes = parse(input);

    // this is still a little bit N^2 log(n), but N is only 1000 so it's fine (24 ms)
    let mut all_edge_lengths: Vec<(usize, usize, u64)> = (0..boxes.len())
        .flat_map(|i| (i + 1..boxes.len()).map(move |j: usize| (i, j)))
        .map(|(i, j)| (i, j, boxes[i].dist_sq(boxes[j])))
        .collect();

    all_edge_lengths.sort_by_key(|(_i, _j, dist)| *dist);

    let mut union_find_tree = UnionFindTree::new(boxes.len());

    for (i, j, _) in all_edge_lengths.iter().copied() {
        union_find_tree.add_edge(i, j);

        if union_find_tree.is_connected() {
            return boxes[i].0 * boxes[j].0;
        }
    }

    unreachable!("Should have eventually gotten connected")
}

#[inline(always)]
fn sq_diff(a: u64, b: u64) -> u64 {
    let diff = a.abs_diff(b);
    diff * diff
}

#[derive(Copy, Clone, Eq, PartialEq, Debug, Hash)]
struct Triple(u64, u64, u64);

impl Triple {
    #[inline(always)]
    fn dist_sq(self, other: Self) -> u64 {
        sq_diff(self.0, other.0) + sq_diff(self.1, other.1) + sq_diff(self.2, other.2)
    }
}

fn parse(input: &str) -> Vec<Triple> {
    input
        .lines()
        .map(|line| {
            let nums: Vec<u64> = line.split(",").map(|token| token.parse::<u64>().unwrap()).collect();
            assert_eq!(nums.len(), 3);
            Triple(nums[0], nums[1], nums[2])
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLE: &str = "162,817,812
57,618,57
906,360,560
592,479,940
352,342,300
466,668,158
542,29,236
431,825,988
739,650,466
52,470,668
216,146,977
819,987,18
117,168,530
805,96,715
346,949,466
970,615,88
941,993,340
862,61,35
984,92,344
425,690,689";

    #[test]
    fn sample_a() {
        assert_eq!(a_with_input(SAMPLE, 10), 5 * 4 * 2);
    }

    #[test]
    fn sample_b() {
        assert_eq!(b_with_input(SAMPLE), 216 * 117);
    }
}
