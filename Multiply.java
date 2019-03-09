import java.io.*;
import java.util.*;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.*;
import org.apache.hadoop.mapreduce.*;
import org.apache.hadoop.mapreduce.lib.input.*;

import org.apache.hadoop.mapreduce.lib.output.*;

/* 
	@Author: Vaibhav Murkute
	Project: Map-Reduce - Matrix-Multiplication
	Date: 02/25/2019
	
*/


class MatrixElement implements Writable {
    	public int tag;
	public int idx;
    	public double value;
	
	MatrixElement () {}

	MatrixElement (int t,int i, double v ) {
        	tag=t; 
        	idx = i;
        	value = v;
    	}

   	 public void write ( DataOutput out ) throws IOException {
        	out.writeInt(tag);
		out.writeInt(idx);
        	out.writeDouble(value);
   	 }

    	public void readFields (DataInput in ) throws IOException {
        	tag= in.readInt();
		idx = in.readInt();
        	value = in.readDouble();
    	}

	@Override
    	public String toString () { 
		return String.valueOf(value); 
	}
 
}


	class JoinKey implements WritableComparable<JoinKey>{
		public int i;
		public int k;
	
		JoinKey() {}
    		JoinKey (int m_row,int n_col) {
     		   i = m_row;
			k = n_col;
    		}

   		 public void write ( DataOutput out ) throws IOException {
			out.writeInt(i);
			out.writeInt(k);
   		 }

   		 public void readFields ( DataInput in ) throws IOException {
       			 i = in.readInt();
			k = in.readInt();
   		 }
		
		@Override
   		 public String toString () { 
			return i+" "+k; 
		}
	
		@Override
		public int compareTo(JoinKey o) {
			if (i == o.i) {
				
				if (k == o.k){
					return 0;
				}else if(k < o.k){
					return -1;
				}else{
					return 1;
				}

			}else if(i < o.i){
				return -1;
			}else{ 
				return 1;
			}
		}
	}

	class ElementProduct implements Writable {
		public double product_val;

		ElementProduct() {}
    		ElementProduct (double value) {
       			product_val = value;
    		}

   		 public void write ( DataOutput out ) throws IOException {
			out.writeDouble(product_val);
   		 }

		public void readFields ( DataInput in ) throws IOException {
      			product_val = in.readDouble();
    		}
		
		@Override
   		 public String toString () {
			return String.valueOf(product_val);
		 }
	}


	public class Multiply {
		public static class M_Mapper extends Mapper<Object,Text,IntWritable,MatrixElement> {
       		 @Override
       		 public void map ( Object key, Text value, Context context )
                        throws IOException, InterruptedException {
           		
			 Scanner s = new Scanner(value.toString()).useDelimiter(",");
            		int m_row = s.nextInt();
			int m_col = s.nextInt();
			double val = s.nextDouble();
            
			MatrixElement m = new MatrixElement(0, m_row, val);
            
			context.write(new IntWritable(m_col), m);
            		s.close();
     	   		}
    		}	


    		public static class N_Mapper extends Mapper<Object,Text,IntWritable,MatrixElement> {
       		 @Override
       		 public void map ( Object key, Text value, Context context )
                	        throws IOException, InterruptedException {
           		 Scanner s = new Scanner(value.toString()).useDelimiter(",");
			int n_row = s.nextInt();
			int n_col = s.nextInt();
			double val = s.nextDouble();
            
			MatrixElement n = new MatrixElement(1, n_col, val);
            	
			context.write(new IntWritable(n_row), n);
           		 s.close();
     		   }
    		}

   		 public static class JoinReducer extends Reducer<IntWritable,MatrixElement,JoinKey,ElementProduct> {
       			static Vector<MatrixElement> m_values = new Vector<MatrixElement>();
        		static Vector<MatrixElement> n_values = new Vector<MatrixElement>();
		
		        @Override
       			 public void reduce ( IntWritable key, Iterable<MatrixElement> values, Context context )
                           throws IOException, InterruptedException {
        			m_values.clear();
        			n_values.clear();
            		
				for (MatrixElement v : values){
               		 		if (v.tag == 0){
					//	m_values.add(v);
						m_values.add(new MatrixElement(0,v.idx, v.value));
					}else {
						n_values.add(new MatrixElement(1, v.idx, v.value));
					//	n_values.add(v);
					}
				}
		 		
				for (MatrixElement m : m_values){
		 			for(MatrixElement n : n_values) {
						context.write(new JoinKey(m.idx, n.idx),new ElementProduct(m.value*n.value));					}
					}				
        			}
    		}		
	
		public static class AggregationMapper extends Mapper<Object,Text,JoinKey,ElementProduct> {
       	 		@Override
		        public void map ( Object key, Text value, Context context )
                	        throws IOException, InterruptedException {
		
		        	StringTokenizer str_tok = new StringTokenizer(value.toString());
        			while (str_tok.hasMoreTokens()) {
		        		int m_row = Integer.parseInt(str_tok.nextToken());
    					int n_col = Integer.parseInt(str_tok.nextToken());
    					double product = Double.parseDouble(str_tok.nextToken());
    			
					context.write(new JoinKey(m_row, n_col), new ElementProduct(product));
        			}
			}
    		}

    		public static class AggregationReducer extends Reducer<JoinKey,ElementProduct,Text,DoubleWritable> {
       			
			@Override
        		public void reduce ( JoinKey key, Iterable<ElementProduct> values, Context context )
                           throws IOException, InterruptedException {
            
				double prod_sum = 0.0;
            			for (ElementProduct p: values){
					prod_sum = prod_sum + p.product_val;
				}
				
				
				context.write(new Text(""+String.valueOf(key.i)+" "+String.valueOf(key.k)),new DoubleWritable(prod_sum));
			}
    	}

	public static void main ( String[] args ) throws Exception {
		Job job1 = Job.getInstance();
        	job1.setJobName("JoinMatrixJob");
        	job1.setJarByClass(Multiply.class);
		job1.setOutputKeyClass(JoinKey.class);
        	job1.setOutputValueClass(ElementProduct.class);
	        job1.setMapOutputKeyClass(IntWritable.class);
        	job1.setMapOutputValueClass(MatrixElement.class);
        	job1.setReducerClass(JoinReducer.class);
               // job1.setInputFormatClass(TextInputFormat.class);
                //job1.setOutputFormatClass(SequenceFileOutputFormat.class);
		MultipleInputs.addInputPath(job1,new Path(args[0]),TextInputFormat.class,M_Mapper.class);
        	MultipleInputs.addInputPath(job1,new Path(args[1]),TextInputFormat.class,N_Mapper.class);
        	FileOutputFormat.setOutputPath(job1,new Path(args[2]));

        	boolean status = job1.waitForCompletion(true);

		if(status){
			Job job2 = Job.getInstance();
				job2.setJobName("AggregateMatrixJob");
				job2.setJarByClass(Multiply.class);
				job2.setOutputKeyClass(Text.class);
				job2.setOutputValueClass(DoubleWritable.class);
				job2.setMapOutputKeyClass(JoinKey.class);
				job2.setMapOutputValueClass(ElementProduct.class);
				job2.setMapperClass(AggregationMapper.class);
				job2.setReducerClass(AggregationReducer.class);
		//		job2.setInputFormatClass(SequenceFileInputFormat.class);
				job2.setOutputFormatClass(TextOutputFormat.class);
				FileInputFormat.setInputPaths(job2,new Path(args[2]));
				FileOutputFormat.setOutputPath(job2,new Path(args[3]));
				status = job2.waitForCompletion(true);
		}

	}


}
