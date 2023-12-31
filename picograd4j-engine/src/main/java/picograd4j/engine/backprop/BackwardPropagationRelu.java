package picograd4j.engine.backprop;

import picograd4j.engine.Value;

public class BackwardPropagationRelu implements BackwardPropagation {
    private Value self;
    private Value out;

    public BackwardPropagationRelu(Value self, Value out) {
        this.self = self;
        this.out = out;
    }

    public void execute() {
        self.grad += (out.data > 0) ? (1 * out.grad) : 0;
    }
}
