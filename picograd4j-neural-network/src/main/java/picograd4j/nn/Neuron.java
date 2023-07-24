package picograd4j.nn;

import java.util.List;
import java.util.ArrayList;
import java.util.Random;

import picograd4j.engine.Value;

public class Neuron extends Module {

    List<Value> w = new ArrayList<>();
    Value b;
    Boolean nonlin = true;
    Random rand = new Random();

    public Neuron(int nin) {
        //Assumes non-linear
        this(nin, true);
    }

    public Neuron(int nin, boolean nonlin){
        this.b = new Value(0);
        this.nonlin = nonlin;
        for (int i = 0; i < nin; i++) {
            float rawValue = (float) (-1 + (1 - (-1)) * this.rand.nextDouble());
            Value value = new Value(rawValue);
            this.w.add(value);
        }
    }

    public Value call(List<Value> x){
        Value act = this.b;
        for (int i = 0; i < this.w.size(); i++) {
            act = act.add(this.w.get(i).multiply(x.get(i)));
        }

        return this.nonlin ? act.relu() : act.relu();
    }
    
    @Override
    public List<Value> parameters(){
        List<Value> parametersList = new ArrayList<>();
        parametersList.addAll(w);
        parametersList.add(b);
        return parametersList;
    }

    @Override
    public String toString(){
        return "{ReLu(" + this.w.size() + ")}";
    }
}
