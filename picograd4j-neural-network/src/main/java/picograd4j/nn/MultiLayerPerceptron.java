package picograd4j.nn;

import java.util.List;
import java.util.ArrayList;

import picograd4j.engine.Value;

public class MultiLayerPerceptron extends Module{
    List<Layer> layers = new ArrayList<>();

    public MultiLayerPerceptron(int nin, List<Integer> nouts){
        int[] sz = new int[nouts.size() + 1];
        sz[0] = nin;
        for (int i = 0; i < nouts.size(); i++) {
            sz[i + 1] = nouts.get(i);
        }

        this.layers = new ArrayList<>();
        for (int i = 0; i < nouts.size(); i++) {
            boolean nonlin = (i != nouts.size() - 1);
            Layer layer = new Layer(sz[i], sz[i + 1], nonlin);
            this.layers.add(layer);
        }
    }

    public List<Value> call(List<Value> x){
        for(Layer layer:layers){
            x = layer.call(x);
        }
        return x;
    }

    @Override
    public List<Value> parameters(){
        List<Value> parameters = new ArrayList<>();
        for (Layer layer : layers) {
            parameters.addAll(layer.parameters());
        }
        return parameters;
    }

    @Override
    public String toString(){
        StringBuilder sb = new StringBuilder();
        sb.append("MLP of [");
        for (int i = 0; i < layers.size(); i++) {
            sb.append(layers.get(i).toString());
            if (i < layers.size() - 1) {
                sb.append(", ");
            }
        }
        sb.append("]");
        return sb.toString();
    }
}
