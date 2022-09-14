use rand::Rng;
use std::mem;
use std::collections::BTreeSet; // Defininetly replace with hashmap
use std::collections::HashSet;
use savefile;
use std::ffi::c_void;
use std::ffi::CString;
use std::os::raw::c_char;

#[macro_use]
extern crate savefile_derive;

// Save structs

#[repr(C)]
#[derive(Savefile)]
struct SaveNeuron {
    val: f32, // The value of the neuron
    off: f32, // The offset to apply to the value when passing it on to the next ones
    x: f32, // The x position of the neuron
    to: Vec<usize>, // The neurons that this neuron feeds into
    from: Vec<usize> // The neurons feed into this neuron
}

impl SaveNeuron {
    fn from(neuron: Neuron) -> SaveNeuron { // Translates an instance of Neuron into an instance of SaveNeuron
        return SaveNeuron {
            val: neuron.val,
            off: neuron.off,
            x: neuron.x,
            to: neuron.to.clone().into_iter().collect::<Vec<usize>>(),
            from: neuron.from.clone().into_iter().collect::<Vec<usize>>()
        };
    }

    fn neuron(&self) -> Neuron { // Translates an instance of SaveNeuron into an instance of Neuron
        return Neuron {
            val: self.val,
            off: self.off,
            x: self.x,
            to: self.to.iter().cloned().collect::<BTreeSet<usize>>(),
            from: self.from.iter().cloned().collect::<BTreeSet<usize>>()
        };
    }
}

#[repr(C)]
#[derive(Savefile)]
struct SaveBrain {
    inputs: Vec<usize>, // The input layer
    activated_neurons: Vec<usize>, // The activated neurons
    neurons: Vec<SaveNeuron>, // The hidden layers which do the actual calculations
    outputs: Vec<usize>, // The output layer
    empty: Vec<usize> // The spots in the brain that are empty
}

impl SaveBrain {
    fn from(brain: Brain) -> SaveBrain { // Translates an instance of Brain into an instance of SaveBrain
        return SaveBrain {
            inputs: brain.inputs.clone().into_iter().collect::<Vec<usize>>(),
            activated_neurons: brain.activated_neurons.clone(),
            neurons: brain.neurons.iter().map(|neuron| SaveNeuron::from(neuron.clone())).collect::<Vec<SaveNeuron>>(),
            outputs: brain.outputs.clone().into_iter().collect::<Vec<usize>>(),
            empty: brain.empty.clone().into_iter().collect::<Vec<usize>>()
        };
    }

    fn brain(&self) -> Brain { // Translates an instance of SaveBrain into an instance of Brain
        return Brain {
            inputs: self.inputs.iter().cloned().collect::<BTreeSet<usize>>(),
            activated_neurons: self.activated_neurons.clone(),
            neurons: self.neurons.iter().map(|neuron| neuron.neuron()).collect::<Vec<Neuron>>(),
            outputs: self.outputs.iter().cloned().collect::<BTreeSet<usize>>(),
            empty: self.empty.iter().cloned().collect::<BTreeSet<usize>>()
        };
    }
}

// Calc structs

#[derive(Clone)]
pub struct Neuron {
    pub val: f32, // The value of the neuron
    pub off: f32, // The offset to apply to the value when passing it on to the next ones
    pub x: f32, // The x position of the neuron
    pub to: BTreeSet<usize>, // The neurons that this neuron feeds into
    pub from: BTreeSet<usize> // The neurons feed into this neuron
}

impl Neuron {
    pub fn new() -> Neuron {
        return Neuron {
            val: 0.0f32,
            off: 1.0f32,
            x: 0.0f32,
            to: BTreeSet::<usize>::new(),
            from: BTreeSet::<usize>::new()
        };
    }
}

#[derive(Clone)]
pub struct Brain {
    pub inputs: BTreeSet<usize>, // The input layer
    pub activated_neurons: Vec<usize>, // The activated neurons
    pub neurons: Vec<Neuron>, // The hidden layers which do the actual calculations
    pub outputs: BTreeSet<usize>, // The output layer
    pub empty: BTreeSet<usize> // The spots in the brain that are empty
}

#[repr(C)]
#[derive(Copy, Clone)]
pub enum Mutation { // The different posible mutations
    AddNeuron, // Add a neuron to the hidden layers 
    RemoveNeuron, // Remove a neuron from the hidden layers
    MutateNeuronOff, // Mutate a neuron's offset
    MutateNeuronX, // Mutate a neuron's x position
    AddNeuronTo, // Add a neuron to the neuron's to
    RemoveNeuronTo, // Remove a neuron from the neuron's from
    Count // The Count enum is to get the count of the number of enums of type Mutations
}

impl Brain {
    fn to_brain_ptr(void_ptr: *mut c_void) -> &'static mut Brain {
        return unsafe { mem::transmute::<*mut c_void, &mut Brain>(void_ptr) };
    }

    fn to_void_ptr(brain_ptr: &mut Brain) -> *mut c_void {
        return unsafe { mem::transmute::<&mut Brain, *mut c_void>(brain_ptr) };
    }

    pub fn new() -> Brain { //Creates a new instance of Brain
        return Brain {
            inputs: BTreeSet::<usize>::new(),
            activated_neurons: Vec::<usize>::new(),
            neurons: Vec::<Neuron>::new(),
            outputs: BTreeSet::<usize>::new(),
            empty: BTreeSet::<usize>::new()
        };
    }

    #[no_mangle]
    pub extern "C" fn new_brain() -> *mut c_void {
        return Brain::to_void_ptr(&mut Brain::new());
    }

    pub fn load(file_name: &str) -> Brain { // Loads a binary file as a brain
        return savefile::prelude::load_file::<SaveBrain, &str>(file_name, 0).unwrap().brain();
    }

    /*
    #[no_mangle]
    pub extern "C" fn brain_load(file_name: &CString) -> *mut c_void {
        return Brain::to_void_ptr(&mut Brain::load(file_name.to_str().unwrap()));
    }
    */

    pub fn save(self, file_name: &str) { // Saves the brain as a binary file
        savefile::prelude::save_file(file_name, 0, &SaveBrain::from(self)).unwrap();
    }

    /*
    #[no_mangle]
    pub extern "C" fn brain_save(self_ptr: *mut c_void, file_name: &CString) {
        Brain::save(Brain::to_brain_ptr(self_ptr).clone(), file_name.to_str().unwrap());
    }
    */

    pub fn get_new_neuron(&mut self) -> usize {
        let mut neuron: usize;
        if self.empty.len() == 0 {
            self.neurons.push(Neuron::new());
            neuron = self.neurons.len() - 1;
        }
        else {
            neuron = 0;
            for empty in self.empty.clone() {
                neuron = empty;
                self.empty.remove(&empty);
                break;
            }
        }
        return neuron;
    }

    #[no_mangle]
    pub extern "C" fn brain_get_new_neuron(self_ptr: *mut c_void) -> usize {
        return Brain::to_brain_ptr(self_ptr).get_new_neuron();
    }

    pub fn add_input(&mut self) { // Adds an input neuron
        let added_input: usize = self.get_new_neuron();
        self.neurons[added_input].x = 0.0;
        self.inputs.insert(added_input);
    }

    #[no_mangle]
    pub extern "C" fn brain_add_input(self_ptr: *mut c_void) {
        Brain::to_brain_ptr(self_ptr).add_input();
    }

    pub fn add_inputs(&mut self, num_inputs: usize) {
        for _ in 0..num_inputs {
            self.add_input();
        }
    }

    #[no_mangle]
    pub extern "C" fn brain_add_inputs(self_ptr: *mut c_void, num_inputs: usize) {
        Brain::to_brain_ptr(self_ptr).add_inputs(num_inputs);
    }

    pub fn add_output(&mut self) { // Adds an output neuron
        let added_output: usize = self.get_new_neuron();
        self.neurons[added_output].x = 1.0;
        self.outputs.insert(added_output);
    }

    #[no_mangle]
    pub extern "C" fn brain_add_output(self_ptr: *mut c_void) {
        Brain::to_brain_ptr(self_ptr).add_output();
    }

    pub fn add_outputs(&mut self, num_outputs: usize) {
        for _ in 0..num_outputs {
            self.add_output();
        }
    }

    #[no_mangle]
    pub extern "C" fn brain_add_outputs(self_ptr: *mut c_void, num_outputs: usize) {
        Brain::to_brain_ptr(self_ptr).add_outputs(num_outputs);
    }

    pub fn add_hidden(&mut self) { // Adds a hidden neuron
        let added_hidden: usize = self.get_new_neuron();
        self.neurons[added_hidden].x = rand::thread_rng().gen_range(0.0f32..1.0f32);
    }

    #[no_mangle]
    pub extern "C" fn brain_add_hidden(self_ptr: *mut c_void) {
        Brain::to_brain_ptr(self_ptr).add_hidden();
    }

    pub fn add_hiddens(&mut self, num_hiddens: usize) {
        for _ in 0..num_hiddens {
            self.add_hidden();
        }
    }

    #[no_mangle]
    pub extern "C" fn brain_add_hiddens(self_ptr: *mut c_void, num_hiddens: usize) {
        Brain::to_brain_ptr(self_ptr).add_hiddens(num_hiddens);
    }

    pub fn remove_neuron(&mut self, neuron: usize) { // Removes a neuron
        for to in &self.neurons.clone()[neuron].to {
            self.neurons[*to].from.remove(&neuron);
        }
        for from in &self.neurons.clone()[neuron].from {
            self.neurons[*from].to.remove(&neuron);
        }
        self.neurons[neuron] = Neuron::new();
        self.empty.insert(neuron);
    }

    #[no_mangle]
    pub extern "C" fn brain_remove_neuron(self_ptr: *mut c_void, neuron: usize) {
        Brain::to_brain_ptr(self_ptr).remove_neuron(neuron);
    }

    pub fn remove_input(&mut self, input: usize) { // Removes an input neuron
        self.remove_neuron(input);
        self.inputs.remove(&input);
    }

    #[no_mangle]
    pub extern "C" fn brain_remove_input(self_ptr: *mut c_void, input: usize) {
        Brain::to_brain_ptr(self_ptr).remove_input(input);
    }
    
    pub fn remove_output(&mut self, output: usize) { // Removes an output neuron
        self.remove_neuron(output);
        self.outputs.remove(&output);
    }

    #[no_mangle]
    pub extern "C" fn brain_remove_output(self_ptr: *mut c_void, output: usize) {
        Brain::to_brain_ptr(self_ptr).remove_output(output);
    }

    pub fn feed_input(&mut self, inputs: Vec<f32>) { // Feeds the input data into the input neurons
        let mut input_loc: usize = 0;
        let input_neurons: Vec<usize> = self.inputs.clone().into_iter().collect();
        for input in inputs {
            self.neurons[input_neurons[input_loc]].val = input;
            input_loc += 1;
        }
    }

    #[no_mangle]
    pub extern "C" fn brain_feed_input(self_ptr: *mut c_void, inputs: *mut f32, len: usize) {
        let mut inputs_vec: Vec<f32> = Vec::new();
        for input in 0..len {
            inputs_vec.push(unsafe { *inputs.add(input) });
        }
        Brain::to_brain_ptr(self_ptr).feed_input(inputs_vec);
    }
    
    pub fn push_input(&mut self) { // Pushes the input neurons to the active neurons vector
        self.activated_neurons.append(&mut self.inputs.clone().into_iter().collect());
    }

    #[no_mangle]
    pub extern "C" fn brain_push_input(self_ptr: *mut c_void) {
        Brain::to_brain_ptr(self_ptr).push_input();
    }

    pub fn get_outputs(&self) -> Vec<f32>{ // Returns the output as a vector of floats
        let mut outputs: Vec<f32> = Vec::new();
        for output in &self.outputs {
            outputs.push(self.neurons[*output].val * self.neurons[*output].off);
        }
        return outputs;
    }

    #[no_mangle]
    pub extern "C" fn brain_get_outputs(self_ptr: *mut c_void) -> *mut f32 {
        return Brain::to_brain_ptr(self_ptr).get_outputs().as_mut_ptr();
    }

    pub fn clear_outputs(&mut self) {
        for output in &self.outputs {
            self.neurons[*output].val = 0.0;
        }
    }

    #[no_mangle]
    pub extern "C" fn brain_clear_outputs(self_ptr: *mut c_void) {
        Brain::to_brain_ptr(self_ptr).clear_outputs();
    }

    pub fn tick(&mut self) { // Runs a full tick of the brain
        let to_remove: usize = self.activated_neurons.len();   
        for neuron in self.activated_neurons.clone() {
            let to_val: f32 = self.neurons[neuron].val * self.neurons[neuron].off;
            if to_val == 0.0 {
                break;
            }
            for to in &self.neurons.clone()[neuron].to {
                self.neurons[*to].val += to_val;
                self.activated_neurons.push(*to);
            }
            if !self.outputs.contains(&neuron) {
                self.neurons[neuron].val = 0.0;
            }
        }
        self.activated_neurons.drain(0..to_remove);
    }

    #[no_mangle]
    pub extern "C" fn brain_tick(self_ptr: *mut c_void) {
        Brain::to_brain_ptr(self_ptr).tick();
    }

    pub fn full_run(&mut self) { // Runs the brain until it finishes
        while self.activated_neurons.len() != 0 {
            self.tick();
        }
    }

    #[no_mangle]
    pub extern "C" fn brain_full_run(self_ptr: *mut c_void) {
        Brain::to_brain_ptr(self_ptr).full_run();
    }

    fn get_mutation(&self) -> Mutation { // Gets the mutation
        unsafe {
            return mem::transmute::<u32, Mutation>(
                rand::thread_rng().gen_range(0..
                    mem::transmute::<Mutation, u32>(Mutation::Count)
                )
            )
        }
    }

    pub fn mutate(&mut self, mutation: Mutation, min_off: f32, max_off: f32) -> bool { // Gets and applies the mutation
        match mutation {
            Mutation::AddNeuron => self.add_hidden(),
            
            Mutation::RemoveNeuron => {
                let neuron: usize = rand::thread_rng().gen_range(0..self.neurons.len());
                if self.inputs.contains(&neuron) || self.outputs.contains(&neuron) || self.empty.contains(&neuron) {
                    return false;
                }
                self.remove_neuron(neuron);
            }

            Mutation::MutateNeuronOff => {
                let neuron: usize = rand::thread_rng().gen_range(0..self.neurons.len());
                self.neurons[neuron].off = rand::thread_rng().gen_range(min_off..max_off);
            }

            Mutation::MutateNeuronX => {
                let neuron: usize = rand::thread_rng().gen_range(0..self.neurons.len());
                if self.inputs.contains(&neuron) || self.outputs.contains(&neuron) || self.empty.contains(&neuron) {
                    return false;
                }
                self.neurons[neuron].x = rand::thread_rng().gen_range(0.0..1.0);
                for to in &self.neurons[neuron].clone().to {
                    if self.neurons[*to].x <= self.neurons[neuron].x {
                        self.neurons[*to].from.remove(&neuron);
                        self.neurons[neuron].to.remove(to);
                    }
                }
                for from in &self.neurons[neuron].clone().from {
                    if self.neurons[*from].x >= self.neurons[neuron].x {
                        self.neurons[*from].to.remove(&neuron);
                        self.neurons[neuron].from.remove(from);
                    }
                }
            }

            Mutation::AddNeuronTo => {
                let hidden_len: usize = self.neurons.len();
                let neuron: usize = rand::thread_rng().gen_range(0..hidden_len);
                if self.outputs.contains(&neuron) || self.empty.contains(&neuron) {
                    return false;
                }
                let to: usize = rand::thread_rng().gen_range(0..hidden_len);
                if self.neurons[neuron].x >= self.neurons[to].x {
                    return false;
                }
                if self.neurons[neuron].to.contains(&to) {
                    return false;
                }
                self.neurons[to].from.insert(neuron);
                self.neurons[neuron].to.insert(to);
            }

            Mutation::RemoveNeuronTo => {
                let neuron: usize = rand::thread_rng().gen_range(0..self.neurons.len());
                if self.outputs.contains(&neuron) || self.neurons[neuron].to.len() == 0 {
                    return false;
                }
                let to_loc: usize = rand::thread_rng().gen_range(0..self.neurons[neuron].to.len());
                let to: usize = self.neurons[neuron].to.clone().into_iter().collect::<Vec<usize>>()[to_loc];
                self.neurons[to].from.remove(&neuron);
                self.neurons[neuron].to.remove(&to);
            }
            
            _ => {
                println!("Invalid mutation!");
                return false;
            }
        }
        return true;
    }

    #[no_mangle]
    pub extern "C" fn brain_mutate(self_ptr: *mut c_void, mutation: Mutation, min_off: f32, max_off: f32) -> bool {
        return Brain::to_brain_ptr(self_ptr).mutate(mutation, min_off, max_off);
    }

    pub fn randomly_mutate(&mut self, min_off: f32, max_off: f32) -> bool {
        return self.mutate(self.get_mutation(), min_off, max_off);
    }

    #[no_mangle]
    pub extern "C" fn brain_randomly_mutate(self_ptr: *mut c_void, min_off: f32, max_off:f32) -> bool {
        return Brain::to_brain_ptr(self_ptr).randomly_mutate(min_off, max_off);
    }
}

#[no_mangle]
pub extern "C" fn test_function() {
    println!("This is from the test function!");
}

#[no_mangle]
pub extern "C" fn test_test_test(testVar: BTreeSet<usize>) {

}

// Everything works except for the CStr, replace those and ur done