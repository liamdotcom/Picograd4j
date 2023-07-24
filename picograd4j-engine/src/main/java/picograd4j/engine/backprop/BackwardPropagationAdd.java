package picograd4j.engine.backprop;

import picograd4j.engine.Value;

public class BackwardPropagationAdd implements BackwardPropagation {
    private Value self;
    private Value other;
    private Value out;

    public BackwardPropagationAdd(Value self, Value other, Value out) {
        this.self = self;
        this.other = other;
        this.out = out;
    }

    public void execute() {
        self.grad += out.grad;
        other.grad += out.grad;
    }
}
