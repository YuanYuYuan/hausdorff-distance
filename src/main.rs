use ndarray::{
    array,
    Array,
    Dimension,
    IntoDimension,
};
use rand::thread_rng;
use rand::seq::SliceRandom;
use std::collections::BinaryHeap;
use std::cmp::Ordering;


fn flatten<D: Dimension>(arr: &Array<bool, D>) -> Vec<D> {
    let mut vec: Vec<_> = arr
        .indexed_iter()
        .filter_map(|(pos, &v)| {
            match v {
                true => Some(pos),
                false => None,
            }
        })
        .map(|x| x.into_dimension())
        .collect();
    vec.shuffle(&mut thread_rng());
    vec
}

fn distance<D: Dimension>(x: &D, y: &D) -> u64 {
    x.slice().iter()
        .zip(y.slice())
        .map(|(x, y)| (*x as i64, *y as i64))
        .fold(0, |sum, (x, y)| sum + (x - y) * (x - y)) as u64
}

fn find_max<D, F>(x_vec: &Vec<D>, y_vec: &Vec<D>, dist: F) -> ((D, D), f64)
where
    D: Dimension + Copy,
    F: Fn(&D, &D) -> u64,
{
    let mut max = u64::MIN;
    let zero = D::zeros(D::NDIM.unwrap());
    let zero_pair = (&zero, &zero);
    let mut pair = zero_pair;

    for x in x_vec {
        let mut min = u64::MAX;
        let mut candidate = zero_pair;
        for y in y_vec {
            let d = dist(x, y);
            if d < max {
                break;
            } else if d < min {
                min = d;
                candidate = (x, y);
            }
        }
        println!("For x: {:?}, min: {:?}, max so far: {:?}", x, min, max);

        if min != u64::MAX && min > max {
            max = min;
            pair = candidate;
        }
    }
    ((*pair.0, *pair.1), (max as f64).sqrt())
}

#[derive(Debug, Default, Copy, Clone, Eq, PartialEq)]
struct Pair<D> {
    x: D,
    y: D,
    v: u64,
}

impl<D: Eq> Ord for Pair<D> {
    fn cmp(&self, other: &Self) -> Ordering {
        other.v.cmp(&self.v)
    }
}

impl<D: Eq> PartialOrd for Pair<D> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        other.v.partial_cmp(&self.v)
    }
}

fn find_max_percentile<D, F>(
    x_vec: &Vec<D>,
    y_vec: &Vec<D>,
    dist: F,
    percentile: f64,
) -> ((D, D), f64)
where
    D: Dimension + Copy,
    F: Fn(&D, &D) -> u64,
{
    assert!(percentile <= 1. && percentile >= 0.);
    let zeros = D::zeros(D::NDIM.unwrap());
    let zeros_pair = (&zeros, &zeros);
    let heap_size = {
        let len = ((x_vec.len() * y_vec.len()) as f64 * (1. - percentile)).ceil() as usize;
        if len > 0 {
            len
        } else {
            1
        }
    };

    let mut heap = BinaryHeap::from(vec![Pair::default(); heap_size]);
    let mut min_of_heap = 0;

    for x in x_vec {
        let mut candidate = zeros_pair;

        let mut min = u64::MAX;
        for y in y_vec {
            let d = dist(x, y);
            if d < min_of_heap {
                break;
            } else if d < min {
                min = d;
                candidate = (x, y);
            }
        }

        let candidate_pair = Pair {
            x: *candidate.0,
            y: *candidate.1,
            v: min,
        };

        if min != u64::MAX {
            let mut heap_changed = false;

            if heap.len() < heap_size {
                heap.push(candidate_pair);
                heap_changed = true
            } else if min > min_of_heap {
                heap.push(candidate_pair);
                heap.pop();
                heap_changed = true
            }

            if heap_changed {
                min_of_heap = heap.peek().unwrap().v;
            }
        }
    }

    let h = heap.peek().unwrap();
    ((h.x, h.y), (h.v as f64).sqrt())
}

fn main() {

    let a = array![
        [0, 0, 1, 1],
        [0, 0, 1, 1],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
    ];
    let b = array![
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [1, 1, 0, 0],
        [1, 1, 0, 0],
    ];

    // let a = array![0, 0, 1, 1];
    // let b = array![1, 1, 0, 0];

    let is_target = |x| x > 0;
    let x_vec = flatten(&a.mapv(is_target));
    let y_vec = flatten(&b.mapv(is_target));

    println!("{:?}", find_max(&x_vec, &y_vec, distance));
    // println!("{:?}", find_max_percentile(&x_vec, &y_vec, distance, 0.95));
}
