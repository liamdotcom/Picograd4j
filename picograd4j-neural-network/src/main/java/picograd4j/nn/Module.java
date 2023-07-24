package picograd4j.nn;

import java.util.ArrayList;
import java.util.List;

import picograd4j.engine.Value;

public class Module {
    public void zeroGradient(){
        for(Value v : parameters()){
            v.grad = 0;
        }
    }

    public List<Value> parameters(){
        return new ArrayList<>();
    }
}
