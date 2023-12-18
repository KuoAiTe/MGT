/**
 * @fileoverview Transformer Visualization D3 javascript code.
 *
 *
 *  Based on: https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/visualization/attention.js
 *
 * Change log:
 *
 * 12/19/18  Jesse Vig   Assorted cleanup. Changed orientation of attention matrices.
 * 12/29/20  Jesse Vig   Significant refactor.
 * 12/31/20  Jesse Vig   Support multiple visualizations in single notebook.
 * 02/06/21  Jesse Vig   Move require config from separate jupyter notebook step
 * 05/03/21  Jesse Vig   Adjust height of visualization dynamically
 * 07/25/21  Jesse Vig   Support layer filtering
 * 03/23/22  Daniel SC   Update requirement URLs for d3 and jQuery (source of bug not allowing end result to be displayed on browsers)
 **/

require.config({
  paths: {
    d3: 'https://cdnjs.cloudflare.com/ajax/libs/d3/5.7.0/d3.min',
    jquery: 'https://cdnjs.cloudflare.com/ajax/libs/jquery/2.0.0/jquery.min',
  }
});
console.log(DATA[0]);
requirejs(['jquery', 'd3'], function ($, d3) {
    const params = {
        'attention': [
            {
                'name': '',
                'attn': Array(0.1, 0.9, 0.9),
                'left_text': Array("A", "B", "B"),
                'right_text': Array("A", "B"),
            }
        ],
        'default_filter': "0",
        'root_div_id': "root",
        'layer': 0,
        'heads': Array(1, 1),
        'include_layers': Array(0, 1, 2),
    };

    const TEXT_SIZE = 32;
    const BOXWIDTH = 400;
    const BOXHEIGHT = 30.5;
    const MATRIX_WIDTH = 115;
    const CHECKBOX_SIZE = 20;
    const TEXT_TOP = 30;

    let headColors;
    try {
        headColors = d3.scaleOrdinal(d3.schemeCategory10);
    } catch (err) {
        headColors = d3.scale.category10();
    }
    let config = {};
    initialize();
    renderVis();
    function initialize() {
        config.data = DATA;
        config.attention = params['attention'];
        config.filter = params['default_filter'];
        config.rootDivId = params['root_div_id'];
        config.nLayers = config.attention[config.filter]['attn'].length;
        config.nHeads = config.attention[config.filter]['attn'][0].length;
        config.layers = params['include_layers']
        config.current_patient_index = 0
        config.current_temporal_index = 0

        if (params['heads']) {
            config.headVis = new Array(config.nHeads).fill(false);
            params['heads'].forEach(x => config.headVis[x] = true);
        } else {
            config.headVis = new Array(config.nHeads).fill(true);
        }
        config.initialTextLength = config.attention[config.filter].right_text.length;
        //config.layer_seq = (params['layer'] == null ? 0 : config.layers.findIndex(layer => params['layer'] === layer));
        config.layer_seq = 0
        config.layer = config.layers[config.layer_seq]

        let layerEl = $(`#${config.rootDivId} #layer`);
        for (const layer of config.layers) {
            layerEl.append($("<option />").val(layer).text(layer));
        }

        let layerPatient = $(`#${config.rootDivId} #patient`);
        let counter = 0;
        for (const user of config.data) {
            layerPatient.append($("<option />").val(counter).text(`${user.original_user_id} ${user.user_id} (${user.labels[0]}) (${user.prediction_scores}, ${user.prediction})` ));
            counter += 1
        }

        layerPatient.on('change', function (e) {
            config.current_patient_index = +e.currentTarget.value;
            config.attention[0] = {
                'attn': [[config.data[config.current_patient_index].user_temporal_attention_layer_0]],
                'left_text': config.data[config.current_patient_index].user_tweet_period.map(element => "TIME_" + element),
                'right_text':  config.data[config.current_patient_index].user_tweet_period.map(element => "TIME_" + element),
            }
            renderVis();
        });
        layerPatient.val(config.current_patient_index).change();

        layerEl.on('change', function (e) {
            config.layer = +e.currentTarget.value;
            config.layer_seq = config.layers.findIndex(layer => config.layer === layer);
            renderVis();
        });
        layerEl.val(config.layer).change();

        let layerTime = $(`#${config.rootDivId} #time`);
        for (const time of [0, 1, 2, 3, 4, 5]) {
            layerTime.append($("<option />").val(time).text(time));
        }
        layerTime.on('change', function (e) {
            config.current_temporal_index = Number(e.currentTarget.value)
            renderVis();
        });
        layerTime.val(config.current_temporal_index).change();



        $(`#${config.rootDivId} #filter`).on('change', function (e) {
            config.filter = e.currentTarget.value;
            renderVis();
        });
    }

    function renderVis() {
        const userIndex = config.current_patient_index
        const temporalIndex = config.current_temporal_index
        const currentData = config.data[userIndex]
        // Load parameters
        const attnData = config.attention[config.filter];
        const leftText = attnData.left_text;
        const rightText = attnData.right_text;

        // Select attention for given layer
        const layerAttention = attnData.attn[config.layer_seq];
        // Clear vis
        $(`#${config.rootDivId} #vis`).empty();
        // Determine size of visualization
        const height = Math.max(leftText.length, rightText.length) * BOXHEIGHT + TEXT_TOP;
        const svg = d3.select(`#${config.rootDivId} #vis`)
            .append('svg')
            .attr("width", "100%")
            .attr("height", height + "px");


        // Display tokens on left and right side of visualization
        //renderText(svg, leftText, true, layerAttention, 0);
        //renderText(svg, rightText, false, layerAttention, MATRIX_WIDTH + BOXWIDTH);

        // Render attention arcs
        //renderAttention(svg, layerAttention);

        // Draw squares at top of visualization, one for each head
        //drawCheckboxes(0, svg, layerAttention);

        //renderUserTweets(`T#${tweet_period_value}: User`, temporal_user_data);
        //renderFriendTweets(`T#${tweet_period_value}: Friend`, temporal_friend_data)
        const d = create_data()
        drawTree(d)

        //drawHeatmap(layerAttention, attnData.left_text, attnData.right_text);
        //drawRelationGraphs(currentData, temporalIndex);
    }
    function truncate(str){
        let n = 1000
        return (str.length > n) ? str.slice(0, n-1) + '...' : str;
      }
    function create_data() {
        const data = {
            "name": "User",
            "layer": -1,
            "children": [

            ]
        }
        for (let i = 0; ; i++ ) {
            let temporal_data = get_temporal_data(i)
            if (temporal_data == undefined || i > 20) break
            temporal_user_data = temporal_data[0]
            temporal_friend_data = temporal_data[1]
            const firstLayerChildren = data["children"]
            firstLayerChildren.push({
                "name": `T#${i}`,
                "layer": 0,
                "i": i,
                "children": [
                    {
                        "name": `T#${i}`,
                        "layer": 1,
                        "children": [

                        {
                            "name": `User`,
                            "layer": 2,
                            "children": [
                                ...(temporal_user_data.map((element, j) => {
                                    console.log(element)
                                    // (${(Number(element.importance)* 100).toFixed(2) }%)
                                    const metric = ''//`(moderate:${Number(element.sentiment['moderate'] * 100).toFixed(2)}%/not depression:${Number(element.sentiment['not depression']* 100).toFixed(2)}%/severe:${Number(element.sentiment['severe']* 100).toFixed(2)}%) `
                                    //:
                                    return {
                                    "name": `[${(Number(element.importance)* 100).toFixed(2)}%] ${j + 1}: ${metric}  ${truncate(element.text)}  `,
                                    "layer": 3,
                                    "importance": element.importance,
                                    "mention": element.mention,
                                    "quote": element.quote,
                                    "reply": element.reply,
                                }})),

                            ]
                        },
                        ...(temporal_friend_data.map(
                            (elements, k) => {
                                return {
                                    "name": `Frnd.${i * elements.length + k}`,
                                    "layer": 2,
                                    "children": elements.map((subelement, l) => {
                                        console.log(subelement)
                                        // 
                                        //(${(Number(subelement.importance)* 100).toFixed(2)}%):
                                        const metric = ``//`(moderate:${Number(subelement.sentiment['moderate'] * 100).toFixed(2)}%/not depression:${Number(subelement.sentiment['not depression']* 100).toFixed(2)}%/severe:${Number(subelement.sentiment['severe']* 100).toFixed(2)}%) `
                                        return {
                                            "name": `[${(Number(subelement.importance)* 100).toFixed(2)}%] ${l + 1}: ${metric} ${truncate(subelement.text)}`,
                                            "layer": 3,
                                            "importance": subelement.importance,
                                            "mention": subelement.mention,
                                            "quote": subelement.quote,
                                            "reply": subelement.reply,
                                        }
                                    })
                                }
                            }
                        ))
                        
                        
                        ]
                    }
                ]
            })
            
        }
        return data
    }
    function get_temporal_data(temporalIndex) {
        if (temporalIndex < 0) return undefined
        const userIndex = config.current_patient_index
        if (userIndex < 0) return undefined
        const currentData = config.data[userIndex]
        tweet_period_value = currentData.user_tweet_period[temporalIndex]
        if (currentData === undefined) return undefined
        if (temporalIndex >= currentData.user_tweets.length) return undefined
        const temporal_user_data = []
        for (let i = 0; i < currentData.user_tweets[temporalIndex].length; i++) {
            temporal_user_data.push({
                text: currentData.user_tweets[temporalIndex][i],
                sentiment: currentData.friend_tweets_sentiments[i],
                importance: currentData.user_tweet_attention_weights[temporalIndex][i],
            })
        }
        const temporal_friend_data = []
        for (let i = 0; i < currentData.friend_tweet_period.length; i++) {
            const friend_data = []
            if (currentData.friend_tweet_period[i] == tweet_period_value) {
                for (let j = 0; j < currentData.friend_tweets[i].length; j++) {
                    friend_data.push({
                        text: currentData.friend_tweets[i][j],
                        sentiment: currentData.friend_tweets_sentiments[i * 2 + j],
                        importance: currentData.friend_tweet_attention_weights[i][j],
                    })
                }
            }
            if (friend_data.length > 0)
                temporal_friend_data.push(friend_data)
        }

        const relation_data = []
        const relations =  ['mention', 'reply', 'quote']
        for (let relation of relations) {
            let gatData = currentData.gat_data[temporalIndex][relation]
            const r = {name: relation, layer:1, children: []}
            if (gatData == undefined || Object.keys(gatData).length == 0) {
            } else {
                const layer_index = 0
                Object.entries(gatData[layer_index][0].attn_data).forEach(([node_id, weight]) => {
                    if (node_id == 0) {
                        for (let i = 0; i < currentData.user_tweets[temporalIndex].length; i++) {
                            temporal_user_data[i][relation] = weight
                        }
                        r['value'] = weight
                    } else {
                        for (let i = 0; i < currentData.friend_tweets[temporalIndex].length; i++) {
                            temporal_friend_data[node_id - 1][i][relation] = weight
                        }
                        r.children.push({
                            name: gatData[layer_index][node_id].name,
                            value: weight,
                        })
                    }
                });

            }
            relation_data.push(r)
        }

        
        return [temporal_user_data, temporal_friend_data]
    }
    function drawTree(data){// Define dimensions
        $("#tree-svg").empty();
        const width = 800;
        const afterWidth = 1800;
        // Create SVG container
      
        const root = d3.hierarchy(data);
        const dx = 28
        const dy = width / (root.height + 8);
        // Create a tree layout.
        const tree = d3.cluster().nodeSize([dx, dy]).separation(function(a, b) {
            return a.parent == b.parent ? 1 : 1.4;
        });;
        tree(root);

        let x0 = Infinity;
        let x1 = -x0;
        root.each(d => {
          if (d.x > x1) x1 = d.x;
          if (d.x < x0) x0 = d.x;
        });
      
        const height = x1 - x0 + dx * 2;
        const svg = d3.select("#tree-svg")
            .attr("width", width)
            .attr("height", height)
            .attr("viewBox", [450, x0 - dx , width, height])
            .attr("style", "max-width: 100%; height: auto; font: 10px sans-serif;");
        
        const layer0Link = []
        const layer1Link = []
        const layer2Link = []
        const layer3Link = []
        
        
        root.links().forEach((d, i) => {
            if (d.source.data.layer == -1 && d.target.data.layer == 0)
                layer0Link.push(d)
            else if (d.source.data.layer == 0 && d.target.data.layer)
                layer1Link.push(d)
            else if (d.target.data.children == undefined)
                layer3Link.push(d)
            else {

                layer2Link.push(d)
            }

        })

        svg.append("g")
            .attr("fill", "none")
            .selectAll()
            .data(layer3Link)
            .enter()
            .append("path")
            .attr("d", d3.linkHorizontal()
                .x(d => d.y/1.5)
                .y(d => d.x/1.5))
            .attr("stroke-opacity", d => 0 + 3 * d.target.data.importance - 0.25)
            .attr("stroke-width", d => 5 * d.target.data.importance)
            .attr("stroke", d => getColorBasedOnImportance(d.target.data.importance));
        drawAttentionLayer0(svg, layer0Link)
        drawAttentionLayer1(svg, layer1Link)
        drawAttentionLayer2(svg, layer2Link)
        const node = svg.append("g")
            .attr("stroke-linejoin", "round")
            .attr("stroke-width", 3)
            .selectAll()
            .data(root.descendants())
            .enter()
            .append("g")
            .attr("transform", d => `translate(${d.y / 1.5},${d.x /1.5})`);

        node.append("circle")
            .attr("fill", d => d.children ? "#555" : "#999")
            .attr("r", 2.5);

        node.append("text")
            .text(d => d.data.name)
            .style("font-size", d => {
                if (d.data.layer == 3) return '18px'
                else if (d.data.layer == 2) return '18px'
                else if (d.data.layer == 1) return '18px'
                else if (d.data.layer == 0) return '18px'
                return  '18px'
            })
            .attr("dy", "0.31em")
            .attr("x", d => {
                if (d.data.layer == 3) return 10
                else if (d.data.layer == 2) return 25
                else if (d.data.layer == 1) return 10
                else if (d.data.layer == 0) return 10
                return  0
            })
            .attr("y", d => {
                if (d.data.layer == 3) return 0
                else if (d.data.layer == 2) return -10
                else if (d.data.layer == 1) return -20
                else if (d.data.layer == 0) return -20
                return  0
            })
            .style("fill", d => {
                let row = d.data
                if (row.layer == 3) {
                    console.log(row.mention, row.quote, row.reply)
                    console.log(d.data, )
                    console.log(Math.abs(row.reply - 0.20717306435108185), Math.abs(row.reply - 0.06411734968423843) < 0.01)
                    if (Math.abs(row.reply - 0.06411734968423843) < 0.0001) return 'black'
                    if (row.mention != undefined && row.mention > 0.2)
                        return  '#bc4749'
                    if (row.reply != undefined && row.reply > 0.2)
                        return  '#bc4749'
                    if (row.quote != undefined && row.quote > 0.2)
                        return  '#bc4749'
                }
                return 'black'
            })
            .attr("text-anchor", d => d.children ? "end" : "start")
           ;
        
        d3.select("#tree-svg")
            .attr("width", `${afterWidth}px`)
    }
    function drawAttentionLayer2(svg, links) {
        const mention_links = []
        const quote_links = []
        const reply_links = []
        links.forEach(d => {
            if (d.target.data.children[0]['mention'] != undefined)
                mention_links.push(d)
            if (d.target.data.children[0]['quote'] != undefined)
                quote_links.push(d)
            if (d.target.data.children[0]['reply'] != undefined)
                reply_links.push(d)
        })
        svg.append("g")
        .attr("fill", "none")
        .selectAll()
        .data(mention_links)
        .enter()
        .append("path")
        .attr("d", d3.linkHorizontal()
            .x(d => d.y/ 1.5)
            .y(d => d.x/ 1.5))
        .attr("stroke-opacity", d => 0.1 + 2.5 * d.target.data.children[0].mention)
        .attr("stroke-width", d => 0.15 + 3 * d.target.data.children[0].mention)
        .attr("stroke", d => {
            return '#ff5714'
        });
        svg.append("g")
        .attr("fill", "none")
        .selectAll()
        .data(reply_links)
        .enter()
        .append("path")
        .attr("d", d3.linkHorizontal()
            .x(d => d.y / 1.5+ 5)
            .y(d => d.x / 1.5))
        .attr("stroke-opacity", d => 0.5 + 2.5 * d.target.data.children[0].reply)
        .attr("stroke-width", d => 0.15 + 3 * d.target.data.children[0].reply)
        .attr("stroke", d => {
            return '#a7c957'
        });
        svg.append("g")
        .attr("fill", "none")
        .selectAll()
        .data(quote_links)
        .enter()
        .append("path")
        .attr("d", d3.linkHorizontal()
            .x(d => d.y/ 1.5 + 10)
            .y(d => d.x/ 1.5))
        .attr("stroke-opacity", d => 0.5 + 2.5 * d.target.data.children[0].quote)
        .attr("stroke-width", d => 0.15 + 3 * d.target.data.children[0].quote)
        .attr("stroke", d => {
            return '#003566'
        });
    }
    function drawAttentionLayer0(svg, links) {
        const offset = 2
        const layerAttention = config.attention[config.filter].attn[config.layer_seq][0];
        const head_index = 0
        const attention_scores = layerAttention[0].slice(offset, offset + links.length);
        const k = svg.append("g")
            .attr("fill", "none")
            .selectAll()
            .data(links)
            .enter()
            .append("path")
            .attr("d", d3.linkHorizontal()
                .x(d => d.y/ 1.5)
                .y(d => d.x/ 1.5))
            .attr("stroke-opacity", d =>{
                const score = attention_scores[d.target.data.i]
                return 0.1 + 3 * score
            })
            .attr("stroke", d => {
                const score = attention_scores[d.target.data.i]
                return getColorBasedOnImportance(score)
            })
            .attr("stroke-width", d =>{
                const score = attention_scores[d.target.data.i]
                return 0.5 + 5 * score
            })


    }
    function drawAttentionLayer1(svg, links) {
        const head_index = 0
        const offset = 2
        const layerAttention = config.attention[config.filter].attn[config.layer_seq][0];
        const attention_scores = [];
        for (let i = 0; i < layerAttention.length; i++) {
            attention_scores.push(
                layerAttention[i].slice(offset, layerAttention.length)
            )
        }
        const left = [];
        const right = []
        links.forEach((data, key) => {
            if (data.source.data.layer == 0 && data.target.data.layer){
                left.push(data.source)
                right.push(data.target)
            }
        })
        const new_links = [];
        for (let i = 0; i < left.length; i++) {
            for (let j = 0; j < right.length; j ++) {
                const counter = i * left.length + j
                new_links.push({
                    source: left[i],
                    target: right[j],
                    i:i,
                    j:j,
                })
            }
        }
        const k = svg.append("g")
            .attr("fill", "none")
            .selectAll()
            .data(new_links)
            .enter()
            .append("path")
            .attr("d", d3.linkHorizontal()
                .x(d => d.y/ 1.5)
                .y(d => d.x/ 1.5))
            .attr("stroke-opacity", d =>{
                const score = attention_scores[d.i][d.j]
                return 0.01 + 3 * score
            })
            .attr("stroke-width", d =>{
                const score = attention_scores[d.i][d.j]
                if (score == undefined) return 0
                return 0.05 + 3 * score
            })
            .attr("stroke", d => {
                const score = attention_scores[d.j][d.i]
                let colorScale = d3.scaleLinear().domain([0, 1]).range(["#bc4749", "#023e8a"]); // Define the corresponding colors
                return getColorBasedOnImportance(score)
            });


    }
    function renderUserTweets(caption, tweetData) {
        $('#user_tweets').empty();
        const container = d3.select("#user_tweets")
        drawTweets(container, caption, tweetData)
    }
    function renderFriendTweets(caption, tweetData) {
        $('#friend_tweets').empty();
        for(let i = 0; i < tweetData.length; i++) {
            const container = d3.select("#friend_tweets")
            drawTweets(container, `${caption} - ${i}`, tweetData[i])
        }

    }
    function drawTweets(container, caption, tweetData) {
        // Create SVG container
        
        container.append("h1")
        .text(caption)
        const svg = container
                    .append('svg')
                    .attr("width", "100%")
                    .attr("height", "200px");
        

        const rectHeight = 20;
        const rectSpacing = 5;
        const textOffsetX = 10; // Adjust the horizontal text offset
        const textOffsetY = rectHeight / 2 + 5; // Adjust the vertical text offset
            
            
        
        const elements = svg
        .selectAll("g")
        .data(tweetData)
        .enter()
        .append("g")
        .attr("transform", (d, i) => `translate(50, ${i * (rectHeight + rectSpacing)})`);

        elements
        .append("rect")
        .attr("width", (d) => d.importance * 300) // Adjust the scale factor as needed
        .attr("height", rectHeight)
        .attr("fill", (d) => getColorBasedOnImportance(d.importance));

        elements
        .append("text")
        .attr("x", textOffsetX)
        .attr("y", textOffsetY)
        .text((d) => d.text)
        .attr("fill", "black") // Adjust the text color as needed
        .style("white-space", "nowrap")
        .style("overflow", "ellipsis")
        .style("overflow", "hidden");
        //style="white-space: nowrap; overflow: hidden; text-overflow: ellipsis;"


    }

    
    function renderText(svg, text, isLeft, attention, leftPos) {

        const textContainer = svg.append("svg:g")
            .attr("id", isLeft ? "left" : "right");

        // Add attention highlights superimposed over words
        textContainer.append("g")
            .classed("attentionBoxes", true)
            .selectAll("g")
            .data(attention)
            .enter()
            .append("g")
            .attr("head-index", (d, i) => i)
            .selectAll("rect")
            .data(d => isLeft ? d : transpose(d)) // if right text, transpose attention to get right-to-left weights
            .enter()
            .append("rect")
            .attr("x", function () {
                var headIndex = +this.parentNode.getAttribute("head-index");
                return leftPos + boxOffsets(headIndex);
            })
            .attr("y", (+1) * BOXHEIGHT)
            .attr("width", BOXWIDTH / activeHeads())
            .attr("height", BOXHEIGHT)
            .attr("fill", function () {
                return headColors(+this.parentNode.getAttribute("head-index"))
            })
            .style("opacity", 0.0);

        const tokenContainer = textContainer.append("g").selectAll("g")
            .data(text)
            .enter()
            .append("g");

        // Add gray background that appears when hovering over text
        tokenContainer.append("rect")
            .classed("background", true)
            .style("opacity", 0.3)
            .attr("fill", "lightgray")
            .attr("x", leftPos)
            .attr("y", (d, i) => TEXT_TOP + i * BOXHEIGHT)
            .attr("width", BOXWIDTH)
            .attr("height", BOXHEIGHT);

        // Add token text
        const textEl = tokenContainer.append("text")
            .text(d => d)
            .attr("font-size", TEXT_SIZE + "px")
            .style("cursor", "default")
            .style("-webkit-user-select", "none")
            .attr("x", leftPos)
            .attr("y", (d, i) => TEXT_TOP + i * BOXHEIGHT);

        if (isLeft) {
            textEl.style("text-anchor", "end")
                .attr("dx", BOXWIDTH - 0.5 * TEXT_SIZE)
                .attr("dy", TEXT_SIZE);
        } else {
            textEl.style("text-anchor", "start")
                .attr("dx", +0.5 * TEXT_SIZE)
                .attr("dy", TEXT_SIZE);
        }

        tokenContainer.on("mouseover", function (d, index) {

            // Show gray background for moused-over token
            textContainer.selectAll(".background")
                .style("opacity", (d, i) => i === index ? 1.0 : 0.0)

            // Reset visibility attribute for any previously highlighted attention arcs
            svg.select("#attention")
                .selectAll("line[visibility='visible']")
                .attr("visibility", null)

            // Hide group containing attention arcs
            svg.select("#attention").attr("visibility", "hidden");

            // Set to visible appropriate attention arcs to be highlighted
            if (isLeft) {
                svg.select("#attention").selectAll("line[left-token-index='" + index + "']").attr("visibility", "visible");
            } else {
                svg.select("#attention").selectAll("line[right-token-index='" + index + "']").attr("visibility", "visible");
            }

            // Update color boxes superimposed over tokens
            const id = isLeft ? "right" : "left";
            const leftPos = isLeft ? MATRIX_WIDTH + BOXWIDTH : 0;
            svg.select("#" + id)
                .selectAll(".attentionBoxes")
                .selectAll("g")
                .attr("head-index", (d, i) => i)
                .selectAll("rect")
                .attr("x", function () {
                    const headIndex = +this.parentNode.getAttribute("head-index");
                    return leftPos + boxOffsets(headIndex);
                })
                .attr("y", (d, i) => TEXT_TOP + i * BOXHEIGHT)
                .attr("width", BOXWIDTH / activeHeads())
                .attr("height", BOXHEIGHT)
                .style("opacity", function (d) {
                    const headIndex = +this.parentNode.getAttribute("head-index");
                    if (config.headVis[headIndex])
                        if (d) {
                            return d[index];
                        } else {
                            return 0.0;
                        }
                    else
                        return 0.0;
                });
        });

        textContainer.on("mouseleave", function () {

            // Unhighlight selected token
            d3.select(this).selectAll(".background")
                .style("opacity", 0.0);

            // Reset visibility attributes for previously selected lines
            svg.select("#attention")
                .selectAll("line[visibility='visible']")
                .attr("visibility", null) ;
            svg.select("#attention").attr("visibility", "visible");

            // Reset highlights superimposed over tokens
            svg.selectAll(".attentionBoxes")
                .selectAll("g")
                .selectAll("rect")
                .style("opacity", 0.0);
        });
    }

    function renderAttention(svg, attention) {

        // Remove previous dom elements
        svg.select("#attention").remove();

        // Add new elements
        svg.append("g")
            .attr("id", "attention") // Container for all attention arcs
            .selectAll(".headAttention")
            .data(attention)
            .enter()
            .append("g")
            .classed("headAttention", true) // Group attention arcs by head
            .attr("head-index", (d, i) => i)
            .selectAll(".tokenAttention")
            .data(d => d)
            .enter()
            .append("g")
            .classed("tokenAttention", true) // Group attention arcs by left token
            .attr("left-token-index", (d, i) => i)
            .selectAll("line")
            .data(d => d)
            .enter()
            .append("line")
            .attr("x1", BOXWIDTH)
            .attr("y1", function () {
                const leftTokenIndex = +this.parentNode.getAttribute("left-token-index")
                return TEXT_TOP + leftTokenIndex * BOXHEIGHT + (BOXHEIGHT / 2)
            })
            .attr("x2", BOXWIDTH + MATRIX_WIDTH)
            .attr("y2", (d, rightTokenIndex) => TEXT_TOP + rightTokenIndex * BOXHEIGHT + (BOXHEIGHT / 2))
            .attr("stroke-width", 2)
            .attr("stroke", function () {
                const headIndex = +this.parentNode.parentNode.getAttribute("head-index");
                return headColors(headIndex)
            })
            .attr("left-token-index", function () {
                return +this.parentNode.getAttribute("left-token-index")
            })
            .attr("right-token-index", (d, i) => i)
        ;
        updateAttention(svg)
    }

    function updateAttention(svg) {
        svg.select("#attention")
            .selectAll("line")
            .attr("stroke-opacity", function (d) {
                const headIndex = +this.parentNode.parentNode.getAttribute("head-index");
                // If head is selected
                if (config.headVis[headIndex]) {
                    // Set opacity to attention weight divided by number of active heads
                    return d / activeHeads()
                } else {
                    return 0.0;
                }
            })
    }

    function boxOffsets(i) {
        const numHeadsAbove = config.headVis.reduce(
            function (acc, val, cur) {
                return val && cur < i ? acc + 1 : acc;
            }, 0);
        return numHeadsAbove * (BOXWIDTH / activeHeads());
    }

    function activeHeads() {
        return config.headVis.reduce(function (acc, val) {
            return val ? acc + 1 : acc;
        }, 0);
    }

    function drawCheckboxes(top, svg) {
        const checkboxContainer = svg.append("g");
        const checkbox = checkboxContainer.selectAll("rect")
            .data(config.headVis)
            .enter()
            .append("rect")
            .attr("fill", (d, i) => headColors(i))
            .attr("x", (d, i) => i * CHECKBOX_SIZE)
            .attr("y", top)
            .attr("width", CHECKBOX_SIZE)
            .attr("height", CHECKBOX_SIZE);

        function updateCheckboxes() {
            checkboxContainer.selectAll("rect")
                .data(config.headVis)
                .attr("fill", (d, i) => d ? headColors(i): lighten(headColors(i)));
        }

        updateCheckboxes();

        checkbox.on("click", function (d, i) {
            if (config.headVis[i] && activeHeads() === 1) return;
            config.headVis[i] = !config.headVis[i];
            updateCheckboxes();
            updateAttention(svg);
        });

        checkbox.on("dblclick", function (d, i) {
            // If we double click on the only active head then reset
            if (config.headVis[i] && activeHeads() === 1) {
                config.headVis = new Array(config.nHeads).fill(true);
            } else {
                config.headVis = new Array(config.nHeads).fill(false);
                config.headVis[i] = true;
            }
            updateCheckboxes();
            updateAttention(svg);
        });
    }


    function lighten(color) {
        const c = d3.hsl(color);
        const increment = (1 - c.l) * 0.6;
        c.l += increment;
        c.s -= increment;
        return c;
    }

    function transpose(mat) {
        return mat[0].map(function (col, i) {
            return mat.map(function (row) {
                return row[i];
            });
        });
    }
    // Function to get color based on importance score
    function getColorBasedOnImportance(importance) {
        let colorScale = d3.scaleLinear().domain([0, 1]).range(["#d90429", "#e5989b"]); // Define the corresponding colors
        return colorScale(importance);
    }
    function drawRelationGraphs(patientData, temporalIndex) {
        const relation_data = []
        const relations =  ['mention', 'reply', 'quote']
        for (let relation of relations) {
            let gatData = patientData.gat_data[temporalIndex][relation]
            const r = {name: relation, layer:1, children: []}
            if (gatData == undefined || Object.keys(gatData).length == 0) {
            } else {
                Object.entries(gatData[0].attn_data).forEach(([node_id, weight]) => {
                    if (node_id == 0) {
                        r['value'] = weight
                    } else {
                        r.children.push({
                            name: gatData[node_id].name,
                            value: weight,
                        })
                    }
                });

            }
            relation_data.push(r)
        }
        
        const drag = (simulation) => {

            function dragstarted(event, d) {
                if (!event.active) simulation.alphaTarget(0.3).restart();
                d.fx = d.x;
                d.fy = d.y;
            }
            
            function dragged(event, d) {
                d.fx = event.x;
                d.fy = event.y;
            }
            
            function dragended(event, d) {
                if (!event.active) simulation.alphaTarget(0);
                d.fx = null;
                d.fy = null;
            }
            
            return d3.drag()
                .on("start", dragstarted)
                .on("drag", dragged)
                .on("end", dragended);
        }
        // Specify the chart’s dimensions.
        const width = 428;
        const height = 400;
        const data = {
            name: "User",
            layer: 0,
            children: relation_data
            
        }
        const factor = 15
        // Compute the graph and start the force simulation.
        const root = d3.hierarchy(data);
        const links = root.links();
        const nodes = root.descendants();

        for (let i = 0;i<nodes.length;i++) {
            if (nodes[i].data.layer == 1) {
                links.push({source:nodes[i], target:nodes[i]})
            }
        }
        const simulation = d3.forceSimulation(nodes)
            .force("link", d3.forceLink(links).id(d => d.id).distance(0).strength(1))
            .force("charge", d3.forceManyBody().strength(-3))
            .force("x", d3.forceX())
            .force("y", d3.forceY());

        // Create the container SVG.
        $("#relation_graphs").empty()
        const svg = d3.select("#relation_graphs")
            .append('svg')
            .attr("width", width)
            .attr("height", height)
            .attr("viewBox", [-width / 2, -height / 2, width, height])
            .attr("style", "max-width: 100%; height: auto; background:orange");

        // Append links.
        const link = svg.append("g")
            .attr("stroke", "#123")
            .attr("stroke-opacity", 1.0)
            .selectAll("line")
            .data(links)
            .enter()
            .append("path")
            .attr("class","link");
            
        // Append nodes.
        const node = svg.append("g")
            .attr("fill", "#fff")
            .attr("stroke", "#000")
            .attr("stroke-width", 1.5)
            .selectAll("circle")
            .data(nodes)
            .enter()
            .append("circle")
            .attr("fill", d => d.children ? null : "#000")
            .attr("stroke", d => d.children ? null : "#fff")
            .attr("r", 3.5)
            .call(drag(simulation));
        var nodelabels = svg.selectAll(".nodelabel")
        .data(nodes)
        .enter()
        .append("text")
        .attr("class", "nodelabel")
        .text(function(d) {
            return d.data.name;
        });

        /*
        var edgelabels = svg.selectAll(".edgelabel")
        .data(links)
        .enter()
        .append('text')
        .style("pointer-events", "none")
        .attr({
            'class': 'edgelabel',
            'id': function(d, i) {
                console.log("?")
            return 'edgelabel' + i
            },
            'font-size': 20,
            'fill': '#aaa'
        })
        
        edgelabels.append("textPath")
        .attr("xlink:href", function(d, i) {
        return '#edge' + i;
        })
        .style("pointer-events", "none")
        .text(function(d, i) {
        return dataset.methlbl[i].name
        });
        
        
        edgelabels.append("textPath")
        .attr("xlink:href", function(d, i) {
            return '#edge' + i;
        })
        .style("pointer-events", "none")
        .text(function(d, i) {
            return dataset.methlbl[i].name
        });*/
          
        simulation.on("tick", () => {
            link.attr("d", function(d) {
                var x1 = d.source.x* factor,
                    y1 = d.source.y* factor,
                    x2 = d.target.x* factor,
                    y2 = d.target.y* factor,
                    dx = x2 - x1,
                    dy = y2 - y1,
                    dr = Math.sqrt(dx * dx + dy * dy),
              
                    // Defaults for normal edge.
                    drx = dr,
                    dry = dr,
                    xRotation = 0, // degrees
                    largeArc = 0, // 1 or 0
                    sweep = 1; // 1 or 0
              
                    // Self edge.
                    if ( x1 === x2 && y1 === y2 ) {
                      // Fiddle with this angle to get loop oriented.
                      xRotation = -45;
              
                      // Needs to be 1.
                      largeArc = 1;
              
                      // Change sweep to change orientation of loop. 
                      //sweep = 0;
              
                      // Make drx and dry different to get an ellipse
                      // instead of a circle.
                      drx = 5;
                      dry = 5;
              
                      // For whatever reason the arc collapses to a point if the beginning
                      // and ending points of the arc are the same, so kludge it.
                      x2 = x2 + 1;
                      y2 = y2 + 1;
                    }
                    return "M" + x1 + "," + y1 + "A" + drx + "," + dry + " " + xRotation + "," + largeArc + "," + sweep + " " + x2 + "," + y2;
                })
            
            link
                .attr("x1", d => d.source.x * factor)
                .attr("y1", d => d.source.y* factor)
                .attr("x2", d => d.target.x* factor)
                .attr("y2", d => d.target.y* factor);
            

            node
                .attr("cx", d => d.x * factor)
                .attr("cy", d => d.y * factor);
            nodelabels
                .attr("x", d => d.x * factor + 10)
                .attr("y", d => d.y * factor + 2)
                
        });

        //invalidation.then(() => simulation.stop());

        //return svg.node();
    }
    function drawHeatmap(attndata, xticklabels, yticklabels){
        xticklabels = ['[CLS]', '[SIAM]', ...xticklabels]
        yticklabels = ['[CLS]', '[SIAM]', ...yticklabels]
        // set the dimensions and margins of the graph
        var margin = {top: 0, right: 0, bottom: 90, left: 20},
          width = 300 - margin.left - margin.right,
          height = 380 - margin.top - margin.bottom;
        console.log(attndata)
        const series = []
        for (let i = 0; i< yticklabels.length; i++) {
            series.push({'key': yticklabels[i], value: attndata[0][i].slice(0, yticklabels.length)})
        }
        const x = d3.scaleBand()
        .domain(xticklabels)
        .range([margin.left, width - margin.right])
        .padding(0.1)
        const y = d3.scaleBand()
        .domain(yticklabels)
        .range([margin.top, height - margin.bottom])
        .padding(0.1)
        const z = d3.scaleSequential(d3.interpolateOrRd)
        .domain([0, d3.max(series, d => d3.max(d.value))])
        // append the svg object to the body of the page
        const legendHeight = 20
        const legendElementWidth = 32
        const legendBins =[
            0,
            0.11240199572344978,
            0.152480399144689956,
            0.20372059871703493,
            0.25496079828937991,
            0.3020099786172488,
            0.5744119743406987,
            0.64868139700641483,
            0.76992159657875983,
          ]
        $("#my_dataviz").empty()
        var svg = d3.select("#my_dataviz")
        .append("svg")
          .attr("width", width + margin.left + margin.right)
          .attr("height", height + margin.top + margin.bottom)
        .append("g")
          .attr("transform",
                "translate(" + margin.left + "," + margin.top + ")");
        // X ticks
        svg.append("g").attr("transform", `translate(${x.bandwidth()/2},${height - margin.bottom})`)
        .call(d3.axisBottom(x).tickSizeOuter(0))
        .call(g => g.select(".domain").remove())
        .selectAll("text")
          .attr("y", 0)
          .attr("x", -9)
          .attr("dy", ".35em")
          .attr("transform", "rotate(270)")
          .style("text-anchor", "end")
          .style("fill", "#777")
        
        // Y ticks
        svg.append("g").attr("transform", `translate(${margin.left},0)`)
        .call(d3.axisLeft(y).tickSize(0).tickPadding(4))
        .call(g => g.select(".domain").remove())
        .selectAll("text")
          .style("fill", "#777")
        
        svg.append("g")
            .selectAll("g")
            .data(series)
            .enter().append("g")
            .attr("transform", d => `translate(0,${y(d.key) + 1})`)
            .append("g")
            .selectAll("rect")
            .data(d => d.value)
            .enter().append("rect")
            .attr("fill"  , d => z(d))
            .attr("x"     , (d,i) => {
                return x(xticklabels[i])
            })
            .attr("y"     , 0)
            .attr("height", y.bandwidth())
            .attr("width" , x.bandwidth())
        const legend = svg.append("g")
            .attr("transform", d => `translate(${margin.left},0)`);
        
        legend
            .selectAll("rect")
            .data(legendBins)
            .enter()
            .append("rect")
            .attr("x", (d, i) => legendElementWidth * i)
            .attr("y", height - (2*legendHeight))
            .attr("width", legendElementWidth)
            .attr("height", legendHeight)
            .style("fill", d => z(d));
        
        legend
            .selectAll("text")
            .data(legendBins)
            .enter()
            .append("text")
            .text(d => "" + (d).toFixed(1))
            .attr("x", (d, i) => legendElementWidth * i)
            .attr("y", height - (legendHeight / 2))
            .style("font-size", "9pt")
            .style("font-family", "Consolas, courier")
            .style("fill", "#aaa");
    }
});