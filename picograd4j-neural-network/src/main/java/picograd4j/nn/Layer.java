package picograd4j.nn;

import java.util.List;
import java.util.ArrayList;

import picograd4j.engine.Value;

public class Layer extends Module {
    List<Neuron> neurons = new ArrayList<>();

    public Layer(int nin, int nout, boolean nonlin){
        for(int i=0; i < nout; i++){
            this.neurons.add(new Neuron(nin, nonlin));
        }
    }

    public List<Value> call(List<Value> x){
        List<Value> out = new ArrayList<>();
        for (Neuron n : neurons) {
            out.add(n.call(x));
        }
        return out;
    }

    @Override
    public List<Value> parameters(){
        List<Value> parameters = new ArrayList<>();
        for (Neuron n : this.neurons) {
            parameters.addAll(n.parameters());
        }
        return parameters;
    }


    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder();
        sb.append("Layer of [");
        for (int i = 0; i < neurons.size(); i++) {
            sb.append(neurons.get(i).toString());
            if (i < neurons.size() - 1) {
                sb.append(", ");
            }
        }
        sb.append("]");
        return sb.toString();
    }
}
