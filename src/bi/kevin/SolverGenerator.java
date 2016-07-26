package bi.kevin;

import com.google.protobuf.TextFormat;

public class SolverGenerator {
    private String train_net, test_net;
    private int snapshot = 1000;
    private String lr_policy = "inv";
    private String snapshot_prefix = "snapshot/neural_net";
    private float gamma = 0.01f;
    private float power = 0.75f;
    private float momentum = 0.9f;
    private float weight_decay = 0.005f;
    private float base_lr = 0.01f;

    public SolverGenerator(String train_net, String test_net){
        this.train_net = train_net;
        this.test_net = test_net;
    }

    public SolverGenerator(String train_net, String test_net, int snapshot,
                           float gamma, float power, float momentum, float weight_decay, float base_lr){
        this.test_net = test_net;
        this.train_net = train_net;
        this.snapshot = snapshot;
        this.gamma = gamma;
        this.power = power;
        this.momentum = momentum;
        this.weight_decay = weight_decay;
        this.base_lr = base_lr;
    }

    public String createSolver(){

        Caffe.SolverParameter.Builder solver = Caffe.SolverParameter.newBuilder();

        solver.setTrainNet(train_net).addTestNet(test_net)
                .setSnapshot(snapshot).setLrPolicy(lr_policy).setSnapshotPrefix(snapshot_prefix).setGamma(gamma)
                .setPower(power).setMomentum(momentum).setWeightDecay(weight_decay).setBaseLr(base_lr);

        return TextFormat.printToString(solver);
    }

}
