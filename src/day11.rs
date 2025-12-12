use ahash::HashMap;
use ahash::HashSet;
use nom::IResult;
use nom::Parser;
use nom::bytes::complete::tag;
use nom::character::complete::alpha1;
use nom::character::complete::space1;
use nom::combinator::eof;
use nom::multi::separated_list1;

const INPUT_FILE: &str = "input/11.txt";

type Node = u32;

pub fn a() -> String {
    let input = std::fs::read_to_string(INPUT_FILE).expect("Input should exist");
    a_with_input(&input).to_string()
}

fn a_with_input(input: &str) -> u64 {
    let graph: Graph = parse(input);

    let output_ind = graph.name_to_idx.get("out").copied().expect("There should an 'out' node");
    let input_ind = graph.name_to_idx.get("you").copied().expect("There should be a 'you' node");

    // Step 2 -- form a topological ordering of the nodes (with start first and out last)
    let sorted_nodes = topological_sort(&graph.edges);

    // Step 3 -- compute the number of paths out of a certain point, into the sink at the end
    find_num_paths(&graph.edges, sorted_nodes, input_ind, output_ind)
}

fn find_num_paths(edges: &[Vec<Node>], mut sorted_nodes: Vec<Node>, source: Node, goal: Node) -> u64 {
    //     This value is 1 at the goal node
    //     If node A connects to B, C, and D, then #(A) = #(B) + #(C) + #(D)
    //     So if we process in sorted order then everything is well-defined, and we get #(start) trivially
    let mut outgoing_paths: HashMap<Node, u64> = HashMap::default();

    // process nodes in reverse order (sinks first, then things that go to that, and so on)
    while let Some(node) = sorted_nodes.pop() {
        if outgoing_paths.contains_key(&node) {
            unreachable!("Cycle detected? Which should be impossible since we sorted, so idk");
        }

        if node == goal {
            outgoing_paths.insert(node, 1);
            continue;
        }

        let mut total = 0;
        for target in edges[node as usize].iter() {
            total += outgoing_paths
                .get(target)
                .copied()
                .expect("Sort should guarantee this number exists");
        }

        outgoing_paths.insert(node, total);
    }

    outgoing_paths.get(&source).copied().unwrap()
}

fn find_reverse_edges(edges: &[Vec<Node>]) -> Vec<HashSet<Node>> {
    let mut reverse_edges: Vec<HashSet<Node>> = vec![HashSet::default(); edges.len()];

    for (source, target) in edges.iter().enumerate() {
        for t in target.iter().copied() {
            reverse_edges[t as usize].insert(source as Node);
        }
    }

    reverse_edges
}

fn topological_sort(edges: &[Vec<Node>]) -> Vec<Node> {
    // Kuhn's algorithm -- iteratively cycle between step A and B, until edges is empty
    //  Step A -- find a node with no incoming edges and add it to the output list (it's first of the rest)
    //  Step B -- remove all connections FROM that node from everything else
    // Repeat until done
    let mut reverse_edges = find_reverse_edges(edges);

    let mut out = Vec::new();
    let mut removed_nodes = HashSet::default();

    let desired_length = reverse_edges.len();

    while out.len() < desired_length {
        // A: find a node with no incoming edges
        let mut node_found = false;

        for i in 0..desired_length {
            if removed_nodes.contains(&i) {
                continue;
            }

            if reverse_edges[i].is_empty() {
                node_found = true;
                removed_nodes.insert(i);
                out.push(i as Node);

                // Step B
                for k in 0..desired_length {
                    reverse_edges[k].remove(&(i as Node));
                }
            }
        }

        if !node_found {
            panic!("Cycle detected (not isolated) -- could not topologically sort");
        }
    }

    out
}

pub fn b() -> String {
    let input = std::fs::read_to_string(INPUT_FILE).expect("Input should exist");
    b_with_input(&input).to_string()
}

fn b_with_input(input: &str) -> u64 {
    // A bit more complicated than part A, but it's fine
    // We want to find all paths that pass from svr to out, passing through dac and fft along the way
    // This is more simply thought of as the sum of all paths
    //      svr->dac->fft->out   +   svr->fft->dac->out
    // We can compute these separately, then (because they are disjoint) we can just add them
    // Also note that since the graph is acyclic, one of those paths is empty (either dac->fft
    // or fft->dac is impossible).

    let graph: Graph = parse(input);

    let output_ind = graph.name_to_idx.get("out").copied().expect("There should an 'out' node");
    let svr_ind = graph.name_to_idx.get("svr").copied().expect("There should be a 'svr' node");
    let dac_ind = graph.name_to_idx.get("dac").copied().expect("There should be a 'dac' node");
    let fft_ind = graph.name_to_idx.get("fft").copied().expect("There should be an 'fft', node");

    // Step 2 -- form a topological ordering of the nodes (with start first and out last)
    let sorted_nodes = topological_sort(&graph.edges);

    // Step 3 -- compute the number of paths out of a certain point, into the sink at the end
    let fft_to_end = find_num_paths(&graph.edges, sorted_nodes.clone(), fft_ind, output_ind);
    let dac_to_fft = find_num_paths(&graph.edges, sorted_nodes.clone(), dac_ind, fft_ind);
    let svr_to_dac = find_num_paths(&graph.edges, sorted_nodes.clone(), svr_ind, dac_ind);

    let dac_to_end = find_num_paths(&graph.edges, sorted_nodes.clone(), dac_ind, output_ind);
    let fft_to_dac = find_num_paths(&graph.edges, sorted_nodes.clone(), fft_ind, dac_ind);
    let svr_to_fft = find_num_paths(&graph.edges, sorted_nodes, svr_ind, fft_ind);

    (svr_to_dac * dac_to_fft * fft_to_end) + (svr_to_fft * fft_to_dac * dac_to_end)
}

struct Graph {
    // map from index to [list of indices it can go to]
    edges: Vec<Vec<Node>>,

    // map from nice name to the node index
    name_to_idx: HashMap<String, Node>,
}

fn parse(input: &str) -> Graph {
    #[derive(Default)]
    struct Interner {
        known: HashMap<String, Node>,
    }

    impl Interner {
        fn next(&mut self, s: &str) -> Node {
            if let Some(val) = self.known.get(s).copied() {
                return val;
            }

            let val = self.known.len() as Node;

            self.known.insert(s.to_string(), val);

            val
        }
    }

    fn parse_fallible(input: &str) -> IResult<&str, Graph> {
        let mut interner = Interner::default();

        let mut edges: Vec<Vec<Node>> = Vec::new();

        for line in input.lines() {
            let (line, source) = alpha1.map(|tok| interner.next(tok)).parse(line)?;
            let (line, _) = tag(": ")(line)?;
            let (line, targets) = separated_list1(space1, alpha1.map(|tok| interner.next(tok))).parse(line)?;
            let (_, _) = eof(line)?;

            while edges.len() <= source as usize {
                edges.push(Vec::new());
            }

            edges[source as usize] = targets;
        }

        while edges.len() < interner.known.len() {
            edges.push(Vec::new()); // just in case the last node is a sink
        }

        let graph = Graph {
            edges,
            name_to_idx: interner.known,
        };

        Ok(("", graph))
    }

    parse_fallible(input).unwrap().1
}

#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLE_A: &str = "aaa: you hhh
you: bbb ccc
bbb: ddd eee
ccc: ddd eee fff
ddd: ggg
eee: out
fff: out
ggg: out
hhh: ccc fff iii
iii: out";

    #[test]
    fn sample_a() {
        assert_eq!(a_with_input(SAMPLE_A), 5);
    }

    const SAMPLE_B: &str = "svr: aaa bbb
aaa: fft
fft: ccc
bbb: tty
tty: ccc
ccc: ddd eee
ddd: hub
hub: fff
eee: dac
dac: fff
fff: ggg hhh
ggg: out
hhh: out";

    #[test]
    fn sample_b() {
        assert_eq!(b_with_input(SAMPLE_B), 2);
    }
}
