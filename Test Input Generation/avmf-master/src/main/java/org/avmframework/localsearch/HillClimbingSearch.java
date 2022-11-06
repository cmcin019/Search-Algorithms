package org.avmframework.localsearch;

import org.avmframework.TerminationException;
import org.avmframework.objective.ObjectiveValue;


public class HillClimbingSearch extends LocalSearch {

    protected ObjectiveValue initial;
    protected ObjectiveValue last;
    protected ObjectiveValue next;
    protected int modifier;
    protected int num;
    protected int dir;

    public HillClimbingSearch() {}

    protected void initialize() throws TerminationException {
        initial = objFun.evaluate(vector);
        modifier = 1;
        num = var.getValue();
      }

    protected void performSearch() throws TerminationException {
        initialize();
        // evaluate left move
        var.setValue(num - modifier);
        ObjectiveValue left = objFun.evaluate(vector);

        // evaluate right move
        var.setValue(num + modifier);
        ObjectiveValue right = objFun.evaluate(vector);

        // find the best direction
        boolean leftBetter = left.betterThan(initial);
        boolean rightBetter = right.betterThan(initial);
      
        if (leftBetter) {
            var.setValue(num - modifier);
        } else if (rightBetter) {
            var.setValue(num + modifier);
        } else {
            var.setValue(num);
        }
    }
}
