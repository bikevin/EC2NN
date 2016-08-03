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

    public EC2Comm(String ec2url, String keyFilePath){

        Security.addProvider(new org.bouncycastle.jce.provider.BouncyCastleProvider());

        sshClient = new SSHClient();
        sshClient.addHostKeyVerifier(new PromiscuousVerifier());

        PKCS8KeyFile keyFile = new PKCS8KeyFile();
        keyFile.init(new File(keyFilePath));

        try {
            sshClient.connect(ec2url);
            sshClient.authPublickey("ubuntu", keyFile);
            session = sshClient.startSession();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public int transferFilesToServer(String[] localFilePaths, String remoteDir){
            try {
                for(String filePath : localFilePaths) {
                    sshClient.newSCPFileTransfer().upload(filePath, remoteDir);
                }
                System.out.println("Transfer Complete");
                return 0;
            } catch (IOException e) {
                e.printStackTrace();
                return -1;
            }

    }

    public String[] trainNet(String localDir, String solverFilePath, int trainIters, int testInterval, int testIters, String[] optional){

        String shellCommand = "python net_trainer.py " + solverFilePath + " " + String.valueOf(trainIters)
                + " " + String.valueOf(testInterval) + " " + String.valueOf(testIters);
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
            transferOutputsToLocal(localDir);
            return outputs;
        } catch (IOException e) {
            e.printStackTrace();
        }
        System.out.println("Training Failed");
        outputs[0] = "Training Failed";
        return outputs;
    }

    public void transferOutputsToLocal(String localDir){
        String[] fileNames = {"train_loss.png", "test_loss.png", "rsquared.png", "snapshot.tar.gz"};
        String compressCommand = "tar -zcvf snapshot.tar.gz snapshot";
        try {
            session = newSession();
            cmd = session.exec(compressCommand);
            System.out.println(IOUtils.readFully(cmd.getErrorStream()).toString());
            System.out.println(IOUtils.readFully(cmd.getInputStream()).toString());
            cmd.join(5, TimeUnit.SECONDS);
            for(String fileName : fileNames) {
                sshClient.newSCPFileTransfer().download(fileName, localDir);
            }
            System.out.println("Download Complete");
        } catch (IOException e) {
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
        String cleanCommand = "find ~/ -maxdepth 1 ! -name \"net_trainer.py\" ! -name \"rsquared.py\" " +
                "! -name \"rsquared.pyc\" ! -name \"csvToh5.py\" -type f -exec rm -f {} \\;";
        String snapCleanCommand = "rm snapshot/*";
        try {
            session = newSession();
            cmd = session.exec(cleanCommand);
            cmd.join(1, TimeUnit.SECONDS);
            session = newSession();
            cmd = session.exec(snapCleanCommand);
            cmd.join(1, TimeUnit.SECONDS);
            System.out.println("Cleaned");
            return 0;
        } catch (IOException e) {
            e.printStackTrace();
        }

        return -1;
    }

    public int csvToH5(String dataCSV, String labelCSV, String h5FileName, String localDir){
        String h5Command = "python csvToh5.py " + dataCSV + " " + labelCSV + " " + h5FileName;
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
