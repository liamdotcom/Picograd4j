package picograd4j;

import java.util.ArrayList;
import java.util.List;

import picograd4j.backprop.BackwardPropagation;
import picograd4j.backprop.BackwardPropagationAdd;
import picograd4j.backprop.BackwardPropagationMultiply;
import picograd4j.backprop.BackwardPropagationPower;
import picograd4j.backprop.BackwardPropagationRelu;

public class Value {
    public float data;
    public float grad = 0;
    public List<Value> prev;
    public BackwardPropagation backprop;

    public Value(float data){
        this.data = data;
        this.prev = null;
    }

    public Value(float data, List<Value> children, BackwardPropagation backprop){
        this.data = data;
        this.prev = children;
        this.backprop = backprop;
    }

    public Value add(int other){
        return add(new Value(other));
    }

    public Value add(float other){
        return add(new Value(other));
    }

    public Value add(double other){
        return add(new Value((float) other));
    }

    public Value add(Value other){
        List<Value> children = new ArrayList<>();
        children.add(this);
        children.add(other);
        
        Value out = new Value(this.data + other.data, children, null);
        out.backprop = new BackwardPropagationAdd(this, other, out);

        return out;
    }

    public Value multiply(int other){
        return multiply(new Value(other));
    }

    public Value multiply(float other){
        return multiply(new Value(other));
    }

    public Value multiply(double other){
        return multiply(new Value((float) other));
    }

    public Value multiply(Value other){
        List<Value> children = new ArrayList<>();
        children.add(this);
        children.add(other);
        
        Value out = new Value(this.data * other.data, children, null);
        out.backprop = new BackwardPropagationMultiply(this, other, out);

        return out;
    }

    public Value power(float other){
        List<Value> children = new ArrayList<>();
        children.add(this);
        
        Value out = new Value((float) Math.pow(this.data, other), children, null);
        out.backprop = new BackwardPropagationPower(this, other, out);

        return out;
    }

    public Value relu(){
        List<Value> children = new ArrayList<>();
        children.add(this);

        float outData = (this.data < 0) ? 0 : this.data;
        Value out = new Value(outData, children, null);
        out.backprop = new BackwardPropagationRelu(this, out);

        return out;
    }

    @Override
    public String toString() {
        return "Value(data={"+ this.data +"}, grad={" +this.grad + "})";
    }
}