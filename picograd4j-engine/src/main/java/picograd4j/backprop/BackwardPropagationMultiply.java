package picograd4j.backprop;

import picograd4j.Value;

public class BackwardPropagationMultiply implements BackwardPropagation {
    private Value self;
    private Value other;
    private Value out;

    public BackwardPropagationMultiply(Value self, Value other, Value out) {
        this.self = self;
        this.other = other;
        this.out = out;
    }

    public void execute() {
        self.grad += other.data * out.grad;
        other.grad += self.data * out.grad;
    }
}