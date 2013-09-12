from html import HTML
import pdb

def create_html_report(topwords,top_index,Pi):
    ht = HTML()
    div = ht.div(align="center")
    heading = div.h2(style="text-align:center")
    subhead = div.h3(style="text-align:center")
    # Creating the header 
    No_of_topics = len(topwords) - 1
    No_of_clusters = top_index.shape[0]
    head = "Topics : %d \n, Cluster : %d" %(No_of_topics,No_of_clusters)
    heading(head)
    
    # Creating the table
    for i in range(top_index.shape[0]):
        head_sub = div.h4(style="text-align:center")
        head_sub("Top Topics in Cluster %d (%.5f)" %(i+1,Pi[i]))
        t = div.table(align="center",border="2",cellpadding="12",cellspacing="0",width="80%")
        tb = t.tbody()
        tr = tb.tr(style="text-align:center")
        tr.th("Alpha")
        tr.th("Top Words")
        ind_alpha_index = top_index[i]
        for index in ind_alpha_index:
            tr = tb.tr(style="text-align:center")
            alpha_val = '%.5f' %index[1]
            tr.td(alpha_val)
            tr.td(topwords[int(index[0])])

    return(ht)
