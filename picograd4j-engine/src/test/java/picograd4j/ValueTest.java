package picograd4j;

import org.junit.Test;
import org.junit.Assert;

public class ValueTest {
    @Test
    public void testSanity() {
        float a = 2;
        float b = -3;

        Value valueA = new Value(a);
        Value valueB = new Value(b);
        Value valueC = valueA.multiply(valueB);
        Value valueD = new Value(10);
        Value valueE = valueC.add(valueD);
        Value valueF = new Value(-2);
        Value valueL = valueE.multiply(valueF);

        valueL.backward();

         Assert.assertEquals(
            6,
            valueA.grad, 
            0.00001f
        );

         Assert.assertEquals(
            -4,
            valueB.grad, 
            0.00001f
        );

        Assert.assertEquals(
            -2,
            valueE.grad, 
            0.00001f
        );

        Assert.assertEquals(
            4,
            valueF.grad,
            0.00001f
        );

        Assert.assertEquals(
            1,
            valueL.grad,
            0.00001f
        );
    }

    @Test
    public void testAdd() {
        float a = 4;
        float b = -7;

        Value valueA = new Value(a);
        Value valueB = new Value(b);
        Value valueC = valueA.add(valueB);

        Assert.assertEquals(
            valueC.data,
            a+b,
            0.00001f
        );
    }

    @Test
    public void testMultiply() {
        float a = 2;
        float b = -3;

        Value valueA = new Value(a);
        Value valueB = new Value(b);
        Value valueC = valueA.multiply(valueB); 

        Assert.assertEquals(
            valueC.data,
            a*b,
            0.00001f
        );
    }

    @Test
    public void testPower() {
        float a = 2;

        Value valueA = new Value(a);
        Value valueB = valueA.power(3); 

        Assert.assertEquals(
            8,
            valueB.data,
            0.00001f
        );
    }

    @Test
    public void testReluPositive() {
        float a = 2;

        Value valueA = new Value(a);
        Value valueB = valueA.relu(); 

        Assert.assertEquals(
            2,
            valueB.data,
            0.00001f
        );
    }

    @Test
    public void testReluNegative() {
        float a = -2;

        Value valueA = new Value(a);
        Value valueB = valueA.relu(); 

        Assert.assertEquals(
            0,
            valueB.data,
            0.00001f
        );
    }
}
