package bi.kevin;

import net.schmizz.sshj.SSHClient;
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

    public void transferFilesToServer(String[] localFilePaths, String remoteDir){
        for(String filePath : localFilePaths){
            try {
                sshClient.newSCPFileTransfer().upload(filePath, remoteDir);
            } catch (IOException e) {
                e.printStackTrace();
            }
        }
    }

    public void trainNet(String localDir, String solverFilePath, int trainIters, int testInterval, int testIters, String[] optional){

        String shellCommand = "net_trainer.py " + solverFilePath + " " + String.valueOf(trainIters)
                + " " + String.valueOf(testInterval) + " " + String.valueOf(testIters);
        for(String string : optional){
            shellCommand += " " + string;
        }

        try {
            cmd = session.exec(shellCommand);
            cmd.join();
            transferOutputsToLocal(localDir);
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public void transferOutputsToLocal(String localDir){
        String[] fileNames = {"train_loss.png", "test_loss.png", "rsquared.png", "snapshot.tar.gz"};
        String compressCommand = "tar -zcvf snapshot.tar.gz";
        try {
            cmd = session.exec(compressCommand);
            cmd.join(5, TimeUnit.SECONDS);
            for(String fileName : fileNames) {
                sshClient.newSCPFileTransfer().download(fileName, localDir);
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public String stopTraining(){
        try {
            session.allocateDefaultPTY();
            cmd.getOutputStream().write(3);
            cmd.getOutputStream().flush();
            cmd.join(1, TimeUnit.SECONDS);
        } catch (IOException e){
            e.printStackTrace();
        }

        return cmd.getExitErrorMessage();
    }

    public void cleanUp(){
        String cleanCommand = "find . -type f -not -name 'net_trainer.py' | xargs -0 rm --";
        String snapCleanCommand = "rm /snapshot/*";
        try {
            cmd = session.exec(cleanCommand);
            cmd.join(1, TimeUnit.SECONDS);
            cmd = session.exec(snapCleanCommand);
            cmd.join(1, TimeUnit.SECONDS);
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    //When this is called, you must declare a new instance of the object to use it
    public void closeConnections(){
        try {
            session.close();
            sshClient.close();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

}
