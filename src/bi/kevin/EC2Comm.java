package bi.kevin;

import net.schmizz.sshj.SSHClient;
import net.schmizz.sshj.common.IOUtils;
import net.schmizz.sshj.connection.ConnectionException;
import net.schmizz.sshj.connection.channel.direct.Session;
import net.schmizz.sshj.transport.TransportException;
import net.schmizz.sshj.transport.verification.PromiscuousVerifier;
import net.schmizz.sshj.userauth.keyprovider.PKCS8KeyFile;

import java.io.File;
import java.io.IOException;
import java.security.Security;
import java.util.concurrent.TimeUnit;

public class EC2Comm {

    SSHClient sshClient;
    Session session;
    Session.Command cmd;
    String userDir;
    String userDirNoSlash;

    public EC2Comm(String ec2url, String keyFilePath, String remoteDir){

        Security.addProvider(new org.bouncycastle.jce.provider.BouncyCastleProvider());

        this.userDir = remoteDir + '/';
        this.userDirNoSlash = remoteDir;

        sshClient = new SSHClient();
        sshClient.addHostKeyVerifier(new PromiscuousVerifier());

        PKCS8KeyFile keyFile = new PKCS8KeyFile();
        keyFile.init(new File(keyFilePath));

        try {
            sshClient.connect(ec2url);
            sshClient.authPublickey("ubuntu", keyFile);
            session = sshClient.startSession();
            cmd = session.exec("mkdir " + remoteDir);
            System.out.println(IOUtils.readFully(cmd.getErrorStream()).toString());
            System.out.println(IOUtils.readFully(cmd.getInputStream()).toString());
            session = newSession();
            cmd = session.exec("mkdir " + remoteDir + "/snapshot");
            System.out.println(IOUtils.readFully(cmd.getErrorStream()).toString());
            System.out.println(IOUtils.readFully(cmd.getInputStream()).toString());
            session = newSession();
            cmd = session.exec("mkdir " + remoteDir + "/image");
            System.out.println(IOUtils.readFully(cmd.getErrorStream()).toString());
            System.out.println(IOUtils.readFully(cmd.getInputStream()).toString());
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public int transferFilesToServer(String[] localFilePaths, String remoteDir){
            try {
                for(String filePath : localFilePaths) {
                    sshClient.newSCPFileTransfer().upload(filePath, userDir + remoteDir);
                }
                System.out.println("Transfer Complete");
                return 0;
            } catch (IOException e) {
                e.printStackTrace();
                return -1;
            }

    }

    public String[] trainNet(String solverFilePath, int trainIters, int testInterval, int testIters, String[] optional){

        String shellCommand = "python net_trainer.py " + solverFilePath + " " + String.valueOf(trainIters)
                + " " + String.valueOf(testInterval) + " " + String.valueOf(testIters) + " " + userDirNoSlash;
        for(String string : optional){
            shellCommand += " " + string;
        }

        System.out.println(shellCommand);

        String[] outputs = new String[2];

        try {
            session = newSession();
            cmd = session.exec(shellCommand);
            outputs[0] = IOUtils.readFully(cmd.getErrorStream()).toString();
            outputs[1] = IOUtils.readFully(cmd.getInputStream()).toString();
            System.out.println(outputs[0]);
            System.out.println(outputs[1]);
            cmd.join();
            return outputs;
        } catch (IOException e) {
            e.printStackTrace();
        }
        System.out.println("Training Failed");
        outputs[0] = "Training Failed";
        return outputs;
    }

    public void transferOutputsToLocal(String localDir){
        String snapCompress = "tar -zcvf " + userDirNoSlash +".tar.gz " + userDirNoSlash;
        try{
            session = newSession();
            cmd = session.exec(snapCompress);
            cmd.join(2, TimeUnit.SECONDS);
        } catch(IOException e){
            e.printStackTrace();
        }
        String[] fileNames = {userDirNoSlash + ".tar.gz"};
        transferFilesToLocal(localDir, fileNames);
    }

    public void analyzeNet(String modelFile, String caffeModelFile, String dataFile){
        try {
            String command = "python net_analyzer.py " + modelFile + " " + caffeModelFile + " " + dataFile + " " + userDirNoSlash;
            session = newSession();
            cmd = session.exec(command);
            cmd.join(5, TimeUnit.SECONDS);
            System.out.println("Analysis Complete");
        } catch(IOException e){
            e.printStackTrace();
        }
    }

    public void predict(String modelFile, String caffeModelFile, String dataFile){
        try{
            String command = "python net_predictor.py " + modelFile + " " + caffeModelFile + " " + dataFile + " " + userDirNoSlash;
            session = newSession();
            cmd = session.exec(command);
            cmd.join(5, TimeUnit.SECONDS);
            System.out.println("Prediction Complete");
        } catch(IOException e){
            e.printStackTrace();
        }
    }

    public void transferFilesToLocal(String localDir, String[] fileNames){
        try {
            for(String fileName : fileNames) {
                sshClient.newSCPFileTransfer().download(fileName, localDir);
            }
            System.out.println("Download Complete");
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public String stopTraining(){
        try {
            session = newSession();
            session.allocateDefaultPTY();
            cmd.getOutputStream().write(3);
            cmd.getOutputStream().flush();
            cmd.join(1, TimeUnit.SECONDS);
        } catch (IOException e){
            e.printStackTrace();
        }

        return cmd.getExitErrorMessage();
    }

    public int cleanUp(){
        String cleanCommand = "rm -rf " + userDirNoSlash;
        String removeGz = "rm -f *.tar.gz";
        try {
            session = newSession();
            cmd = session.exec(cleanCommand);
            cmd.join(1, TimeUnit.SECONDS);
            session = newSession();
            cmd = session.exec(removeGz);
            cmd.join(1, TimeUnit.SECONDS);
            System.out.println("Cleaned");
            return 0;
        } catch (IOException e) {
            e.printStackTrace();
        }

        return -1;
    }

    public int csvToH5(String dataCSV, String labelCSV, String h5FileName, String localDir){
        String h5Command = "python csvToh5.py " + userDir + dataCSV + " " + userDir + labelCSV + " " + h5FileName;
        String[] outputs = {h5FileName};
        try {
            session = newSession();
            cmd = session.exec(h5Command);
            cmd.join(1, TimeUnit.SECONDS);
            transferFilesToLocal(localDir, outputs);
            return 0;
        } catch (IOException e){
            e.printStackTrace();
        }
        return -1;
    }

    public int drawNet(String netProtoFile, String imageName) {
        String imageCommand = "python caffe-master/python/draw_net.py " + userDir + netProtoFile + " " + userDir + "image/" + imageName;
        try {
            session = newSession();
            cmd = session.exec(imageCommand);
            System.out.println(IOUtils.readFully(cmd.getErrorStream()).toString());
            System.out.println(IOUtils.readFully(cmd.getInputStream()).toString());
            cmd.join(5, TimeUnit.SECONDS);
            System.out.println("Drawing Complete");
            return 0;
        } catch(IOException e){
            e.printStackTrace();
        }

        return -1;
    }

    //When this is called, you must declare a new instance of the object to use it
    public int closeConnections(){
        try {
            session.close();
            sshClient.close();
            return 0;
        } catch (IOException e) {
            e.printStackTrace();
        }

        return -1;
    }

    public Session newSession() throws ConnectionException, TransportException {
        return sshClient.startSession();
    }

}
