package picograd4j.backprop;

import picograd4j.Value;

public class BackwardPropagationPower implements BackwardPropagation {
    private Value self;
    private float other;
    private Value out;

    public BackwardPropagationPower(Value self, float other, Value out) {
        this.self = self;
        this.other = other;
        this.out = out;
    }

    public void execute() {
        self.grad += (other * Math.pow(self.data, (other - 1))) * out.grad;
    }
}