//! Hidden markov models!
//!

use std::collections::HashMap;
use std::marker::Copy;
use std::cmp::Eq;
use std::hash::Hash;

extern crate regex;
use regex::Regex;

struct HMM<X: Eq + Hash + Copy, Y: Eq + Hash + Copy> {
    hidden:     HashMap<X, f64>,
    observed:   HashMap<X, HashMap<Y, f64>>,
    hidden_transition: HashMap<X, HashMap<X, f64>>,
}

impl<X: Eq + Hash + Copy, Y: Eq + Hash + Copy> HMM<X, Y> {
    pub fn new() -> HMM<X, Y> {
        HMM {
            hidden: HashMap::new(),
            observed: HashMap::new(),
            hidden_transition: HashMap::new()
        }
    }

    pub fn train(&mut self, input: &[(X, Y)]) {

        // collect data
        let mut prev_hidden: Option<X> = None;
        for v in input.iter() {
            // get hidden data
            let occ = self.hidden.entry(v.0).or_insert(0f64);
            *occ += 1f64;
 
            // get transition of hidden -> hidden
            match prev_hidden {
                Some(ph) => {
                    let hid_trans_occ = self.hidden_transition.entry(ph).or_insert(HashMap::new());

                    let trans_occ = hid_trans_occ.entry(v.0).or_insert(0f64);
                    *trans_occ += 1f64;
                },
                None => {
                     let hid_trans_occ = self.hidden_transition.entry(v.0).or_insert(HashMap::new());
                   
                }
            }

            // get  observed data
            let hid_occ = self.observed.entry(v.0).or_insert({
                let mut tmp = HashMap::new();
                tmp.insert(v.1, 0f64);
                tmp
            });
            
            let obsv_occ = hid_occ.entry(v.1).or_insert(0f64);
            *obsv_occ += 1f64;
            prev_hidden = Some(v.0.clone());
        }

        //average the values of the data
        //
        for (hid, oh_hash) in self.observed.iter_mut() {
            if self.hidden.contains_key(hid) {
                let num = *self.hidden.get(hid).unwrap();
                for (_, oh_val) in oh_hash.iter_mut() {
                    *oh_val /= num;
                }
            }
        }
        
        for (hid, oth_hash) in self.hidden_transition.iter_mut() {
            let num = oth_hash.values().fold(0f64, |acc, &i| acc + i);
            for (_, oth_val) in oth_hash.iter_mut() {
                *oth_val /= num;
            }
        }
        
        let num_hid = self.hidden.values().fold(0f64, |acc, &i| acc + i);

        for (_, val) in self.hidden.iter_mut() {
            *val /= num_hid;
        }
    }

    pub fn get_hidden(&self) -> &HashMap<X, f64> { &self.hidden } 
    pub fn get_observed(&self) -> &HashMap<X, HashMap<Y, f64>> { &self.observed } 
    pub fn get_hidden_transition(&self) -> &HashMap<X, HashMap<X, f64>> { &self.hidden_transition } 

    pub fn prob_of_hid(&self, hid: X) -> Option<f64> { 
        match self.hidden.get(&hid) {
            Some(val) =>  Some(*val),
            None => None
        }
    }
    pub fn prob_of_ob_hid(&self, ob: Y, hid: X) -> Option<f64> {
        match self.observed.get(&hid) {
            Some(oh_hash) => {
                match oh_hash.get(&ob) {
                    Some(val) => Some(*val),
                    None => None
                }
            },
            None => None
        }
    }
    pub fn prob_of_hid_ob(&self, hid: X, ob: Y) -> f64 {
        let poh = self.prob_of_ob_hid(ob, hid).unwrap_or(0f64);
        let ph = self.prob_of_hid(hid).unwrap_or(0f64);
        let mut d: f64 = 0f64;

        for hid in self.hidden.iter() {
            d += *hid.1 * self.prob_of_ob_hid(ob, *hid.0).unwrap_or(0f64);
        }

        (poh * ph) / d
    }
}

fn parser(input: &str) -> Vec<(&str, &str)> {
    let mut res = Vec::new();
    let re = Regex::new(r"([\S]+)/([\S]+)").unwrap();
    for cap in re.captures_iter(input) {
        res.push((
                cap.at(1).unwrap(),
                cap.at(2).unwrap()
                ));
    }
    res
}
fn main() {
    let test = "
        s/w s/s s/s s/s s/c s/c s/c s/c s/c s/f
        h/w h/w h/w h/w h/s h/s h/s h/s h/c h/c
        h/w h/w h/w h/w h/s h/s h/s h/s h/c h/c
        h/w h/w h/w h/w h/s h/s h/s h/s h/c h/c
        h/w h/w h/w h/w h/s h/s h/s h/s h/c h/c
        ".to_string();

    let mut hmm = HMM::new();
    hmm.train(&parser(&test));

    println!("{:#?}", hmm.get_hidden());
    println!("{:#?}", hmm.get_observed());
    println!("{:#?}", hmm.get_hidden_transition());
    println!("{}", hmm.prob_of_hid_ob("h", "w"));

    println!("Hello, world!");
}
