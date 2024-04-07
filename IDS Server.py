from scapy.all import sniff, IP, TCP, Raw
import joblib
import pandas as pd
import time
from sklearn.preprocessing import StandardScaler, LabelEncoder

def le(df):
    for col in df.columns:
        if df[col].dtype == 'object':
                label_encoder = LabelEncoder()
                df[col] = label_encoder.fit_transform(df[col])

clf = joblib.load('decision_tree_model_2.joblib')
scale = StandardScaler()
selected_features = ['protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes', 'count', 'same_srv_rate', 'dst_host_srv_count', 'dst_host_same_srv_rate', 'dst_host_same_src_port_rate']
#cap = pd.DataFrame(columns=['protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes', 'count', 'same_srv_rate', 'dst_host_srv_count', 'dst_host_same_srv_rate', 'dst_host_same_src_port_rate'])


# define a callback function to process the captured packet
def process_packet(packet):
    packet.show()
    if packet.haslayer(IP) and packet.haslayer(TCP):
            try:
                TCPpkt = packet[TCP]
                load=TCPpkt.load.decode('utf-8')
                load = load.split(',')
                row = pd.DataFrame([load], columns=['protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes', 'count', 'same_srv_rate', 'dst_host_srv_count', 'dst_host_same_srv_rate', 'dst_host_same_src_port_rate'])
                df=pd.read_csv('capture.csv')
                df = df.append(row, ignore_index=True)
                #save the data to a csv file
                df.to_csv('capture.csv', index=False)
                print(df)   
                le(df)
                df = scale.fit_transform(df)
                print('prediction:',clf.predict(df))
                new_data = pd.read_csv("capture.csv")
                new_data = new_data[selected_features]
                le(new_data)
                #print the 3rd row
                new_data = scale.fit_transform(new_data)
                predictions=clf.predict(new_data)
                print(predictions)
            except:
                pass

    
        

# start sniffing packets and pass them to the callback function
sniff(filter="ip src 192.168.0.1 and tcp port 1234", prn=process_packet)
